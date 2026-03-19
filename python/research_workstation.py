import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def _safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _clip01(value):
    return max(0.0, min(1.0, _safe_float(value)))


@dataclass
class ResearchConfig:
    slippage_bps: float = 8.0
    commission_bps: float = 2.0
    trade_delay_days: int = 1
    min_holding_days: int = 3
    long_entry: float = 0.60
    long_exit: float = 0.52
    short_entry: float = 0.40
    short_exit: float = 0.48
    max_exposure: float = 1.0


class QuantResearchWorkstation:
    def __init__(self, config=None):
        self.config = config or ResearchConfig()

    def run(self, symbol, period, optimize_strategy=True):
        df_raw = get_stock_data(ticker=symbol, period=period)
        return self.run_with_raw_data(symbol, period, df_raw, optimize_strategy)

    def run_with_raw_data(self, symbol, period, df_raw, optimize_strategy=True):
        data_quality = self.evaluate_data_quality(df_raw, symbol, period)
        df_features = self.build_feature_store(df_raw)
        feature_bundle = self.prepare_feature_matrix(df_features)
        models = self.run_model_stack(feature_bundle)
        signals = self.build_signal_stack(df_features, models)
        validation = self.run_validation(feature_bundle, signals)
        strategy = self.run_strategy(df_features, signals, validation, optimize_strategy)
        charts = self.build_charts(df_features, signals, strategy)
        summary = self.build_summary(symbol, period, df_features, data_quality, signals, validation, strategy)
        explanations = self.build_explanations(summary, signals, validation, strategy, data_quality)

        return {
            "summary": summary,
            "data_quality": data_quality,
            "signals": signals,
            "models": models,
            "validation": validation,
            "strategy": strategy,
            "charts": charts,
            "explanations": explanations,
        }

    def evaluate_data_quality(self, df, symbol, period):
        date_series = pd.to_datetime(df["Date"])
        duplicate_dates = int(date_series.duplicated().sum())
        missing_ratio = float(df[["Open", "High", "Low", "Close", "Volume"]].isna().mean().mean())
        close_flat = int((df["Close"].diff().fillna(0) == 0).sum())
        low_history = len(df) < 180
        score = 100
        score -= min(25, duplicate_dates * 5)
        score -= min(25, int(missing_ratio * 400))
        score -= 15 if low_history else 0
        score -= min(10, close_flat // 10)
        score = max(20, score)

        issues = []
        if duplicate_dates:
            issues.append(f"{duplicate_dates} duplicate dates detected")
        if missing_ratio > 0:
            issues.append("Missing OHLCV values were cleaned before research")
        if low_history:
            issues.append("Limited history reduces walk-forward depth")
        if close_flat > max(5, len(df) * 0.08):
            issues.append("High number of unchanged closes may indicate sparse trading")
        if not issues:
            issues.append("No material data quality alerts")

        return {
            "symbol": symbol,
            "period": period,
            "canonical_schema": ["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"],
            "rows": int(len(df)),
            "date_start": date_series.iloc[0].strftime("%Y-%m-%d"),
            "date_end": date_series.iloc[-1].strftime("%Y-%m-%d"),
            "duplicate_dates": duplicate_dates,
            "missing_value_ratio": round(missing_ratio, 6),
            "flat_close_days": close_flat,
            "score": int(score),
            "quality_state": "robust" if score >= 80 else "usable" if score >= 60 else "fragile",
            "issues": issues,
            "cache_key": f"{symbol}:{period}:{int(date_series.iloc[-1].timestamp())}",
        }

    def build_feature_store(self, df):
        base = add_technical_indicators(df.copy())
        base["Log_Return"] = np.log(base["Close"]).diff().fillna(0)
        base["Gap_Return"] = ((base["Open"] / base["Close"].shift(1)) - 1).replace([np.inf, -np.inf], 0).fillna(0)
        base["Intra_Range"] = ((base["High"] - base["Low"]) / base["Close"]).replace([np.inf, -np.inf], 0).fillna(0)
        base["Drawdown_20d"] = (base["Close"] / base["Close"].rolling(20).max() - 1).fillna(0)
        base["Realized_Vol_20d"] = base["Log_Return"].rolling(20).std().mul(np.sqrt(252)).fillna(0)
        base["Realized_Vol_60d"] = base["Log_Return"].rolling(60).std().mul(np.sqrt(252)).fillna(0)
        base["Vol_Regime"] = (base["Realized_Vol_20d"] / base["Realized_Vol_60d"].replace(0, np.nan)).replace([np.inf, -np.inf], 1).fillna(1)
        base["Range_Expansion"] = (base["Intra_Range"] / base["Intra_Range"].rolling(20).mean().replace(0, np.nan)).replace([np.inf, -np.inf], 1).fillna(1)
        base["Volume_Regime"] = (base["Volume"] / base["Volume"].rolling(20).mean().replace(0, np.nan)).replace([np.inf, -np.inf], 1).fillna(1)
        base["Trend_20d"] = base["Close"].pct_change(20).fillna(0)
        base["Trend_60d"] = base["Close"].pct_change(60).fillna(0)
        base["Downside_Vol_20d"] = base["Log_Return"].clip(upper=0).rolling(20).std().mul(np.sqrt(252)).fillna(0)
        base["Dividend_Event"] = (base["Dividends"] > 0).astype(int)
        base["Split_Event"] = (base["Stock Splits"] > 0).astype(int)
        base["Close_Z20"] = (
            (base["Close"] - base["Close"].rolling(20).mean()) /
            base["Close"].rolling(20).std().replace(0, np.nan)
        ).replace([np.inf, -np.inf], 0).fillna(0)

        rolling_cols = ["Log_Return", "Gap_Return", "Intra_Range", "Volume_Regime", "Trend_20d", "Trend_60d"]
        for col in rolling_cols:
            norm_col = f"{col}_norm"
            base[norm_col] = (
                (base[col] - base[col].rolling(60).mean()) /
                base[col].rolling(60).std().replace(0, np.nan)
            ).replace([np.inf, -np.inf], 0).fillna(0)

        base["Target"] = (base["Close"].shift(-1) > base["Close"]).astype(int)
        base = base.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        return base

    def prepare_feature_matrix(self, df):
        exclude = {"Date", "Target"}
        features = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]
        X = df[features].copy()
        y = df["Target"].astype(int).copy()
        return {"df": df, "X": X, "y": y, "features": features}

    def build_regime_snapshot(self, df):
        latest = df.iloc[-1]
        trend = _safe_float(latest.get("Trend_60d", 0))
        vol_regime = _safe_float(latest.get("Vol_Regime", 1))
        adx = _safe_float(latest.get("ADX", 20))
        if vol_regime > 1.25:
            regime = "high_volatility"
        elif trend > 0.06 and adx > 22:
            regime = "trending_up"
        elif trend < -0.06 and adx > 22:
            regime = "trending_down"
        else:
            regime = "ranging"

        raw = {
            "trending_up": max(0.01, trend + 0.15),
            "trending_down": max(0.01, -trend + 0.15),
            "ranging": max(0.01, 1.0 - abs(trend) * 5 - abs(vol_regime - 1)),
            "high_volatility": max(0.01, vol_regime - 0.7),
        }
        total = sum(raw.values())
        probs = {k: round(v / total, 4) for k, v in raw.items()}
        return regime, probs

    def run_model_stack(self, bundle):
        df = bundle["df"]
        X = bundle["X"]
        y = bundle["y"]
        n = len(df)
        if n < 90:
            latest_up = float(y.tail(min(len(y), 20)).mean()) if len(y) else 0.5
            regime, regime_probs = self.build_regime_snapshot(df)
            return {
                "base_models": {
                    "fallback": {
                        "latest_probability_up": round(latest_up, 4),
                        "latest_vote": "up" if latest_up >= 0.5 else "down",
                        "test_accuracy": 0.5,
                        "sample_size": int(len(y)),
                    }
                },
                "ensemble_probability_up": round(latest_up, 4),
                "feature_importance": [],
                "regime_classifier": {
                    "current_regime": regime,
                    "probabilities": regime_probs,
                },
            }
        split_idx = max(int(n * 0.75), min(180, n - 30))
        split_idx = min(max(split_idx, 80), n - 20)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        models = {
            "gradient_boosting": GradientBoostingClassifier(random_state=42, n_estimators=160, learning_rate=0.05, max_depth=2),
            "random_forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=6, min_samples_leaf=4),
            "logistic": LogisticRegression(max_iter=400),
        }

        model_outputs = {}
        latest_vector = X.iloc[[-1]]
        probabilities = []

        for name, estimator in models.items():
            estimator.fit(X_train, y_train)
            calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
            calibrated.fit(X_train, y_train)
            proba_test = calibrated.predict_proba(X_test)[:, 1]
            latest_proba = _clip01(calibrated.predict_proba(latest_vector)[0][1])
            pred_test = (proba_test >= 0.5).astype(int)
            accuracy = accuracy_score(y_test, pred_test)
            probabilities.append(latest_proba)

            model_outputs[name] = {
                "latest_probability_up": round(latest_proba, 4),
                "latest_vote": "up" if latest_proba >= 0.5 else "down",
                "test_accuracy": round(float(accuracy), 4),
                "sample_size": int(len(y_test)),
            }

        ensemble_probability = float(np.mean(probabilities))
        regime, regime_probs = self.build_regime_snapshot(df)

        feature_importance = []
        gb = models["gradient_boosting"]
        if hasattr(gb, "feature_importances_"):
            importance_pairs = list(zip(bundle["features"], gb.feature_importances_))
            importance_pairs.sort(key=lambda item: item[1], reverse=True)
            for feature, score in importance_pairs[:10]:
                feature_importance.append({"feature": feature, "importance": round(float(score), 4)})

        return {
            "base_models": model_outputs,
            "ensemble_probability_up": round(ensemble_probability, 4),
            "feature_importance": feature_importance,
            "regime_classifier": {
                "current_regime": regime,
                "probabilities": regime_probs,
            },
        }

    def build_signal_stack(self, df, models):
        latest = df.iloc[-1]
        obv_anchor = max(1.0, abs(_safe_float(df["OBV"].tail(60).mean(), 1)))
        sleeves = {
            "momentum": np.tanh((_safe_float(latest.get("Trend_20d")) * 8) + (_safe_float(latest.get("MACD_hist")) * 3)),
            "trend": np.tanh((_safe_float(latest.get("ADX")) - 20) / 12 + (_safe_float(latest.get("Trend_60d")) * 6)),
            "mean_reversion": np.tanh((-_safe_float(latest.get("Close_Z20")) * 0.9) + ((50 - _safe_float(latest.get("RSI"), 50)) / 25)),
            "breakout": np.tanh((_safe_float(latest.get("Range_Expansion"), 1) - 1) * 2 + (_safe_float(latest.get("Volume_Regime"), 1) - 1)),
            "volatility": np.tanh((_safe_float(latest.get("Vol_Regime"), 1) - 1) * -2 + (_safe_float(latest.get("Realized_Vol_20d"), 0) - _safe_float(latest.get("Realized_Vol_60d"), 0)) * -1.5),
            "volume_pressure": np.tanh((_safe_float(latest.get("Relative_Volume"), 1) - 1) * 1.5 + (_safe_float(latest.get("OBV")) / obv_anchor)),
            "downside_risk": np.tanh((_safe_float(latest.get("Downside_Vol_20d"), 0) - _safe_float(latest.get("Realized_Vol_20d"), 0)) * -2 + (_safe_float(latest.get("Drawdown_20d"), 0) * 5)),
        }
        sleeves = {key: round(float(np.clip(value, -1, 1)), 4) for key, value in sleeves.items()}

        raw_model_score = models["ensemble_probability_up"] * 2 - 1
        regime = models["regime_classifier"]["current_regime"]
        regime_fit = {
            "trending_up": {"momentum": 1.2, "trend": 1.2, "breakout": 1.0},
            "trending_down": {"downside_risk": 1.2, "trend": 0.8, "mean_reversion": 0.7},
            "ranging": {"mean_reversion": 1.2, "volatility": 1.0},
            "high_volatility": {"volatility": 1.3, "downside_risk": 1.1, "breakout": 0.9},
        }.get(regime, {})

        weighted = {name: value * regime_fit.get(name, 1.0) for name, value in sleeves.items()}
        sleeve_contribution = sum(weighted.values()) / max(len(weighted), 1)
        composite = np.clip(((raw_model_score * 0.45) + (sleeve_contribution * 0.55) + 1) * 50, 0, 100)
        conviction = composite / 100.0
        position_state = "long" if conviction >= self.config.long_entry else "short" if conviction <= self.config.short_entry else "flat"
        risk_state = "elevated" if regime == "high_volatility" else "normal"

        sleeve_rows = []
        for name, value in sleeves.items():
            sleeve_rows.append({
                "name": name,
                "score": value,
                "direction": "bullish" if value > 0.15 else "bearish" if value < -0.15 else "neutral",
                "strength": round(abs(value), 4),
                "decay": round(max(0.05, 1 - abs(value) * 0.6), 4),
            })
        sleeve_rows.sort(key=lambda item: item["strength"], reverse=True)

        return {
            "composite_index_latest": round(float(composite), 2),
            "ensemble_conviction": round(float(conviction), 4),
            "position_state": position_state,
            "risk_state": risk_state,
            "alpha_sleeves": sleeve_rows,
            "model_score": round(float(raw_model_score), 4),
            "confidence_bands": {
                "short_entry": self.config.short_entry,
                "short_exit": self.config.short_exit,
                "long_exit": self.config.long_exit,
                "long_entry": self.config.long_entry,
            },
        }

    def walk_forward_predictions(self, bundle, signals):
        X = bundle["X"].reset_index(drop=True)
        y = bundle["y"].reset_index(drop=True)
        df = bundle["df"].reset_index(drop=True)
        predictions = []
        weights_history = []

        for test_start in range(140, len(df) - 40, 20):
            test_end = min(test_start + 40, len(df) - 1)
            X_train = X.iloc[:test_start]
            y_train = y.iloc[:test_start]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            if len(X_train) < 80 or len(X_test) < 10:
                continue

            local_models = {
                "gb": GradientBoostingClassifier(random_state=42, n_estimators=120, learning_rate=0.05, max_depth=2),
                "rf": RandomForestClassifier(random_state=42, n_estimators=120, max_depth=6, min_samples_leaf=4),
                "lr": LogisticRegression(max_iter=300),
            }
            fold_probs = {}
            metrics = {}
            for name, estimator in local_models.items():
                estimator.fit(X_train, y_train)
                prob = estimator.predict_proba(X_test)[:, 1]
                fold_probs[name] = prob
                metrics[name] = accuracy_score(y_test, (prob >= 0.5).astype(int))

            accuracy_sum = sum(metrics.values()) or 1.0
            model_weights = {name: score / accuracy_sum for name, score in metrics.items()}
            weights_history.append(model_weights)

            for idx, local_idx in enumerate(range(test_start, test_end)):
                ensemble_prob = sum(model_weights[name] * fold_probs[name][idx] for name in fold_probs)
                ensemble_prob = np.clip(ensemble_prob + (signals["model_score"] * 0.01), 0, 1)
                predictions.append({
                    "date": df["Date"].iloc[local_idx].strftime("%Y-%m-%d"),
                    "probability_up": float(ensemble_prob),
                    "prediction": int(ensemble_prob >= 0.5),
                    "actual": int(y.iloc[local_idx]),
                })

        return predictions, weights_history

    def run_validation(self, bundle, signals):
        predictions, weights_history = self.walk_forward_predictions(bundle, signals)
        if not predictions:
            return {
                "walk_forward": {"anchored_accuracy": 0, "rolling_accuracy": 0, "folds": 0, "latest_confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}},
                "regime_slices": [],
                "calibration": [],
                "benchmarks": {},
                "weight_stability": {"dominant_model": "n/a", "dispersion": 0},
            }

        pred_df = pd.DataFrame(predictions)
        anchored_accuracy = (pred_df["prediction"] == pred_df["actual"]).mean()
        rolling_accuracy = (pred_df.tail(min(60, len(pred_df)))["prediction"] == pred_df.tail(min(60, len(pred_df)))["actual"]).mean()

        calibration = []
        pred_df["bucket"] = pd.cut(pred_df["probability_up"], bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0], include_lowest=True)
        for bucket, group in pred_df.groupby("bucket", observed=False):
            calibration.append({
                "bucket": str(bucket),
                "avg_probability_up": round(float(group["probability_up"].mean()), 4),
                "actual_up_rate": round(float(group["actual"].mean()), 4),
                "count": int(len(group)),
            })

        df = bundle["df"].set_index(bundle["df"]["Date"].dt.strftime("%Y-%m-%d"))
        regime_rows = []
        for name in ["trending_up", "trending_down", "ranging", "high_volatility"]:
            if name == "trending_up":
                mask = (df["Trend_60d"] > 0.06) & (df["ADX"] > 22)
            elif name == "trending_down":
                mask = (df["Trend_60d"] < -0.06) & (df["ADX"] > 22)
            elif name == "high_volatility":
                mask = df["Vol_Regime"] > 1.25
            else:
                mask = ~(((df["Trend_60d"] > 0.06) & (df["ADX"] > 22)) | ((df["Trend_60d"] < -0.06) & (df["ADX"] > 22)) | (df["Vol_Regime"] > 1.25))
            dates = set(df[mask].index.tolist())
            regime_pred = pred_df[pred_df["date"].isin(dates)]
            if len(regime_pred) == 0:
                continue
            regime_rows.append({
                "regime": name,
                "accuracy": round(float((regime_pred["prediction"] == regime_pred["actual"]).mean()), 4),
                "sample_size": int(len(regime_pred)),
            })

        dominant_counts = {}
        dispersions = []
        for row in weights_history:
            dominant = max(row, key=row.get)
            dominant_counts[dominant] = dominant_counts.get(dominant, 0) + 1
            dispersions.append(float(np.std(list(row.values()))))
        dominant_model = max(dominant_counts, key=dominant_counts.get) if dominant_counts else "n/a"

        actual = pred_df["actual"].astype(int)
        pred = pred_df["prediction"].astype(int)
        tp = int(((pred == 1) & (actual == 1)).sum())
        tn = int(((pred == 0) & (actual == 0)).sum())
        fp = int(((pred == 1) & (actual == 0)).sum())
        fn = int(((pred == 0) & (actual == 1)).sum())

        return {
            "walk_forward": {
                "anchored_accuracy": round(float(anchored_accuracy), 4),
                "rolling_accuracy": round(float(rolling_accuracy), 4),
                "folds": int(len(weights_history)),
                "latest_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            },
            "regime_slices": regime_rows,
            "calibration": calibration,
            "benchmarks": {
                "buy_hold_hit_rate": round(float(bundle["y"].mean()), 4),
                "naive_momentum_hit_rate": round(float((bundle["df"]["Trend_20d"] > 0).astype(int).eq(bundle["y"]).mean()), 4),
            },
            "weight_stability": {
                "dominant_model": dominant_model,
                "dispersion": round(float(np.mean(dispersions) if dispersions else 0), 4),
            },
        }

    def run_strategy(self, df, signals, validation, optimize_strategy):
        df = df.copy().reset_index(drop=True)
        composite = []
        for _, row in df.iterrows():
            momentum = np.tanh(_safe_float(row.get("Trend_20d")) * 8)
            trend = np.tanh((_safe_float(row.get("ADX")) - 20) / 12 + (_safe_float(row.get("Trend_60d")) * 6))
            mean_reversion = np.tanh((-_safe_float(row.get("Close_Z20")) * 0.9) + ((50 - _safe_float(row.get("RSI"), 50)) / 25))
            breakout = np.tanh((_safe_float(row.get("Range_Expansion"), 1) - 1) * 2 + (_safe_float(row.get("Volume_Regime"), 1) - 1))
            volatility = np.tanh((_safe_float(row.get("Vol_Regime"), 1) - 1) * -2)
            raw_score = np.mean([momentum, trend, mean_reversion, breakout, volatility])
            composite.append(np.clip((raw_score + 1) * 50, 0, 100))
        df["Composite_Index"] = composite
        df["Probability_Up"] = df["Composite_Index"] / 100

        position = 0
        hold_days = 0
        positions = []
        target_weights = []
        trade_log = []
        entry_price = None
        entry_date = None
        slippage = self.config.slippage_bps / 10000
        commission = self.config.commission_bps / 10000

        for idx in range(len(df)):
            prob = _safe_float(df.at[idx, "Probability_Up"], 0.5)
            desired = position
            if position == 0:
                if prob >= self.config.long_entry:
                    desired = 1
                elif prob <= self.config.short_entry:
                    desired = -1
            elif position == 1 and hold_days >= self.config.min_holding_days and prob <= self.config.long_exit:
                desired = 0
            elif position == -1 and hold_days >= self.config.min_holding_days and prob >= self.config.short_exit:
                desired = 0

            if desired != position:
                if position != 0 and entry_price is not None:
                    exit_price = _safe_float(df.at[idx, "Close"])
                    gross = ((exit_price / entry_price) - 1) * position
                    net = gross - slippage - commission
                    trade_log.append({
                        "entry_date": entry_date,
                        "exit_date": df.at[idx, "Date"].strftime("%Y-%m-%d"),
                        "side": "long" if position == 1 else "short",
                        "return_pct": round(net * 100, 2),
                        "holding_days": hold_days,
                    })
                if desired != 0:
                    entry_price = _safe_float(df.at[idx, "Close"])
                    entry_date = df.at[idx, "Date"].strftime("%Y-%m-%d")
                    hold_days = 0
                else:
                    entry_price = None
                    entry_date = None
                    hold_days = 0
            else:
                hold_days = hold_days + 1 if desired != 0 else 0

            position = desired
            positions.append(position)
            strength = abs(prob - 0.5) * 2
            target_weights.append(round(min(self.config.max_exposure, strength), 4) if position != 0 else 0.0)

        df["Position"] = positions
        df["Target_Weight"] = target_weights
        df["Market_Return"] = df["Close"].pct_change().fillna(0)
        executed_position = df["Position"].shift(self.config.trade_delay_days).fillna(0)
        cost = executed_position.diff().abs().fillna(0) * (slippage + commission)
        df["Strategy_Return"] = (executed_position * df["Target_Weight"] * df["Market_Return"]) - cost
        df["BuyHold_Value"] = (1 + df["Market_Return"]).cumprod()
        df["Strategy_Value"] = (1 + df["Strategy_Return"]).cumprod()
        df["Strategy_Drawdown"] = df["Strategy_Value"] / df["Strategy_Value"].cummax() - 1

        total_return = _safe_float(df["Strategy_Value"].iloc[-1] - 1)
        benchmark_return = _safe_float(df["BuyHold_Value"].iloc[-1] - 1)
        turnover = _safe_float(df["Position"].diff().abs().sum())
        sharpe_like = _safe_float(df["Strategy_Return"].mean() / (df["Strategy_Return"].std() + 1e-9) * np.sqrt(252))
        tail_loss = _safe_float(df["Strategy_Return"].quantile(0.05))
        exposure = float(df["Position"].abs().mean())
        avg_holding = float(np.mean([trade["holding_days"] for trade in trade_log])) if trade_log else 0.0
        wins = [trade for trade in trade_log if trade["return_pct"] > 0]
        losses = [trade for trade in trade_log if trade["return_pct"] <= 0]
        latest_prob = _safe_float(df["Probability_Up"].iloc[-1], 0.5)
        current_regime = "high_volatility" if _safe_float(df["Vol_Regime"].iloc[-1], 1) > 1.25 else "trend" if abs(_safe_float(df["Trend_60d"].iloc[-1], 0)) > 0.06 else "range"

        return {
            "execution_config": {
                "slippage_bps": self.config.slippage_bps,
                "commission_bps": self.config.commission_bps,
                "trade_delay_days": self.config.trade_delay_days,
                "min_holding_days": self.config.min_holding_days,
                "max_exposure": self.config.max_exposure,
                "mode": "adaptive_research" if optimize_strategy else "baseline_research",
            },
            "latest_state": {
                "target_state": "long" if positions[-1] == 1 else "short" if positions[-1] == -1 else "flat",
                "conviction": round(latest_prob, 4),
                "position_size_score": round(abs(latest_prob - 0.5) * 2, 4),
                "expected_horizon_days": 10 if current_regime == "range" else 20,
                "entry_exit_rationale": "Conviction band crossed with hysteresis-aware execution controls",
                "risk_flags": [
                    "High volatility regime" if current_regime == "high_volatility" else "Normal realized volatility",
                    "Turnover elevated" if turnover > 15 else "Turnover controlled",
                ],
            },
            "performance": {
                "strategy_total_return": round(total_return * 100, 2),
                "buy_hold_total_return": round(benchmark_return * 100, 2),
                "alpha_vs_buy_hold": round((total_return - benchmark_return) * 100, 2),
                "max_drawdown": round(_safe_float(df["Strategy_Drawdown"].min()) * 100, 2),
                "tail_loss_5pct": round(tail_loss * 100, 2),
                "turnover": round(turnover, 2),
                "gross_exposure": round(exposure, 4),
                "avg_holding_days": round(avg_holding, 2),
                "trade_count": len(trade_log),
                "win_rate": round((len(wins) / len(trade_log) * 100) if trade_log else 0, 2),
                "expectancy": round(np.mean([trade["return_pct"] for trade in trade_log]) if trade_log else 0, 2),
                "payoff_ratio": round((np.mean([trade["return_pct"] for trade in wins]) / abs(np.mean([trade["return_pct"] for trade in losses]))) if wins and losses and np.mean([trade["return_pct"] for trade in losses]) != 0 else 0, 2),
                "sharpe_like": round(sharpe_like, 2),
            },
            "trade_log": trade_log[-12:],
            "chart_data": {
                "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "strategy": [_safe_float(v) for v in df["Strategy_Value"]],
                "buyhold": [_safe_float(v) for v in df["BuyHold_Value"]],
                "exposure": [_safe_float(v) for v in df["Target_Weight"]],
            },
            "diagnostics": {
                "walk_forward_accuracy": validation["walk_forward"]["rolling_accuracy"],
                "dominant_model": validation["weight_stability"]["dominant_model"],
                "regime_fit": signals["risk_state"],
            },
        }

    def build_charts(self, df, signals, strategy):
        tail = df.tail(120).copy()
        if "Composite_Index" not in tail.columns:
            tail["Composite_Index"] = 50.0
        strategy_tail = strategy.get("chart_data")
        if strategy_tail:
            strategy_dates = strategy_tail["dates"]
            strategy_map = {date: idx for idx, date in enumerate(strategy_dates)}
            tail["Strategy_Value"] = [
                strategy_tail["strategy"][strategy_map.get(date.strftime("%Y-%m-%d"), 0)]
                for date in tail["Date"]
            ]
            tail["BuyHold_Value"] = [
                strategy_tail["buyhold"][strategy_map.get(date.strftime("%Y-%m-%d"), 0)]
                for date in tail["Date"]
            ]
            tail["Target_Weight"] = [
                strategy_tail["exposure"][strategy_map.get(date.strftime("%Y-%m-%d"), 0)]
                for date in tail["Date"]
            ]
        else:
            tail["Strategy_Value"] = (1 + tail["Close"].pct_change().fillna(0)).cumprod()
            tail["BuyHold_Value"] = tail["Strategy_Value"]
            tail["Target_Weight"] = 0.0
        return {
            "price": {
                "dates": tail["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "close": [round(_safe_float(v), 4) for v in tail["Close"]],
                "ma20": [round(_safe_float(v), 4) for v in tail["MA20"]],
                "ma50": [round(_safe_float(v), 4) for v in tail["MA50"]],
                "drawdown": [round(_safe_float(v) * 100, 4) for v in tail["Drawdown_20d"]],
            },
            "composite": {
                "dates": tail["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "values": [round(_safe_float(v), 4) for v in tail["Composite_Index"]],
                "long_entry": [signals["confidence_bands"]["long_entry"] * 100] * len(tail),
                "long_exit": [signals["confidence_bands"]["long_exit"] * 100] * len(tail),
                "short_entry": [signals["confidence_bands"]["short_entry"] * 100] * len(tail),
                "short_exit": [signals["confidence_bands"]["short_exit"] * 100] * len(tail),
            },
            "strategy": {
                "dates": tail["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "strategy": [round((_safe_float(v) - 1) * 100, 4) for v in tail["Strategy_Value"]],
                "buyhold": [round((_safe_float(v) - 1) * 100, 4) for v in tail["BuyHold_Value"]],
                "exposure": [round(_safe_float(v), 4) for v in tail["Target_Weight"]],
            },
            "alpha_heatmap": signals["alpha_sleeves"],
        }

    def build_summary(self, symbol, period, df, data_quality, signals, validation, strategy):
        last = df.iloc[-1]
        upside = signals["ensemble_conviction"]
        recommendation = "Accumulate long bias" if upside >= self.config.long_entry else "Lean short tactically" if upside <= self.config.short_entry else "Stay neutral / wait"
        return {
            "symbol": symbol,
            "period": period,
            "last_close": round(_safe_float(last["Close"]), 4),
            "day_change_pct": round(_safe_float(last["Price_Change_1d"]) * 100, 2),
            "recommendation": recommendation,
            "target_state": strategy["latest_state"]["target_state"],
            "current_regime": validation["weight_stability"]["dominant_model"] + " / " + signals["risk_state"],
            "ensemble_conviction": round(upside, 4),
            "composite_index": signals["composite_index_latest"],
            "research_score": round((signals["ensemble_conviction"] * 40) + (data_quality["score"] * 0.35) + (validation["walk_forward"]["rolling_accuracy"] * 25), 2),
        }

    def build_explanations(self, summary, signals, validation, strategy, data_quality):
        active = signals["alpha_sleeves"][:3]
        active_text = ", ".join([f"{item['name']} ({item['direction']})" for item in active]) if active else "no dominant sleeves"
        return {
            "overview": f"{summary['symbol']} is in a {strategy['latest_state']['target_state']} posture with composite conviction {signals['composite_index_latest']:.1f}/100.",
            "why_now": f"Primary drivers are {active_text}. Walk-forward rolling accuracy is {validation['walk_forward']['rolling_accuracy']:.2%} and data quality is {data_quality['quality_state']}.",
            "risk": f"Risk state is {signals['risk_state']} with tail loss estimate {strategy['performance']['tail_loss_5pct']}% and max drawdown {strategy['performance']['max_drawdown']}%.",
            "execution": f"Execution uses next-bar simulation with {self.config.slippage_bps} bps slippage, {self.config.commission_bps} bps commission, and minimum holding period of {self.config.min_holding_days} days.",
        }


def run_quant_research(symbol, period="2y", optimize_strategy=True):
    workstation = QuantResearchWorkstation()
    result = workstation.run(symbol=symbol, period=period, optimize_strategy=optimize_strategy)
    return json.dumps(result)


async def run_quant_research_async(symbol, period="2y", optimize_strategy=True):
    workstation = QuantResearchWorkstation()
    df_raw = await get_stock_data_async(ticker=symbol, period=period)
    result = workstation.run_with_raw_data(
        symbol=symbol,
        period=period,
        df_raw=df_raw,
        optimize_strategy=optimize_strategy,
    )
    return json.dumps(result)
