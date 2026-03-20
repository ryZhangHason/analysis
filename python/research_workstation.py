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
        self.stage_cache = {}

    def get_stage(self, key, builder):
        if key not in self.stage_cache:
            self.stage_cache[key] = builder()
        return self.stage_cache[key]

    def run(self, symbol, period, optimize_strategy=True, execution_mode="quick"):
        df_raw = get_stock_data(ticker=symbol, period=period)
        return self.run_with_raw_data(symbol, period, df_raw, optimize_strategy, execution_mode)

    def run_with_raw_data(self, symbol, period, df_raw, optimize_strategy=True, execution_mode="quick"):
        data_quality = self.evaluate_data_quality(df_raw, symbol, period)
        df_features = self.get_stage("feature_store", lambda: self.build_feature_store(df_raw))
        feature_bundle = self.get_stage("model_matrix", lambda: self.prepare_feature_matrix(df_features))
        models = self.get_stage("model_stack", lambda: self.run_model_stack(feature_bundle, execution_mode))
        signals = self.get_stage("signals", lambda: self.build_signal_stack(df_features, models))
        validation = self.get_stage("validation", lambda: self.run_validation(feature_bundle, execution_mode))
        strategy = self.get_stage(
            "strategy_ranking",
            lambda: self.run_strategy_ranking(df_features, signals, validation, optimize_strategy),
        )
        charts = self.get_stage("charts", lambda: self.build_charts(df_features, signals, strategy))
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
            "runtime": {
                "execution_mode": execution_mode,
                "cached_stages": sorted(list(self.stage_cache.keys())),
            },
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
        warnings = []
        if duplicate_dates:
            issues.append(f"{duplicate_dates} duplicate dates detected")
        if missing_ratio > 0:
            issues.append("Missing OHLCV values were cleaned before research")
        if low_history:
            warnings.append("Limited history reduces confidence and strategy ranking depth")
        if close_flat > max(5, len(df) * 0.08):
            warnings.append("Sparse trading characteristics detected")
        if not issues and not warnings:
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
            "warnings": warnings,
            "minimum_sample_warning": low_history,
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
        base["Baseline_Momentum_Signal"] = np.where(base["Trend_20d"] > 0, 1, 0)

        rolling_cols = ["Log_Return", "Gap_Return", "Intra_Range", "Volume_Regime", "Trend_20d", "Trend_60d"]
        for col in rolling_cols:
            norm_col = f"{col}_norm"
            base[norm_col] = (
                (base[col] - base[col].rolling(60).mean()) /
                base[col].rolling(60).std().replace(0, np.nan)
            ).replace([np.inf, -np.inf], 0).fillna(0)

        base["Target"] = (base["Close"].shift(-1) > base["Close"]).astype(int)
        return base.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    def prepare_feature_matrix(self, df):
        exclude = {"Date", "Target"}
        features = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]
        X = df[features].copy()
        y = df["Target"].astype(int).copy()
        n = len(df)
        train_end = max(90, int(n * 0.55))
        valid_end = max(train_end + 30, int(n * 0.75))
        valid_end = min(valid_end, n - 20)
        return {"df": df, "X": X, "y": y, "features": features, "splits": {"train_end": train_end, "valid_end": valid_end}}

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
        probabilities = {k: round(v / total, 4) for k, v in raw.items()}
        return regime, probabilities

    def run_model_stack(self, bundle, execution_mode):
        df = bundle["df"]
        X = bundle["X"]
        y = bundle["y"]
        splits = bundle["splits"]
        train_end = splits["train_end"]
        valid_end = splits["valid_end"]

        if len(df) < 100:
            regime, probs = self.build_regime_snapshot(df)
            latest_up = float(y.tail(min(len(y), 20)).mean()) if len(y) else 0.5
            return {
                "base_models": {
                    "fallback": {
                        "latest_probability_up": round(latest_up, 4),
                        "latest_vote": "up" if latest_up >= 0.5 else "down",
                        "validation_accuracy": 0.5,
                        "test_accuracy": 0.5,
                    }
                },
                "ensemble_probability_up": round(latest_up, 4),
                "feature_importance": [],
                "regime_classifier": {"current_regime": regime, "probabilities": probs},
                "confidence_governance": {"state": "insufficient_evidence", "minimum_sample_warning": True},
            }

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_valid, y_valid = X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]
        X_test, y_test = X.iloc[valid_end:], y.iloc[valid_end:]

        model_defs = {
            "gradient_boosting": GradientBoostingClassifier(random_state=42, n_estimators=140 if execution_mode == "quick" else 180, learning_rate=0.05, max_depth=2),
            "random_forest": RandomForestClassifier(random_state=42, n_estimators=120 if execution_mode == "quick" else 220, max_depth=6, min_samples_leaf=4),
            "logistic": LogisticRegression(max_iter=400),
        }

        base_models = {}
        probabilities = []
        feature_importance = []
        latest_vector = X.iloc[[-1]]

        for name, estimator in model_defs.items():
            estimator.fit(X_train, y_train)
            calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
            calibrated.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
            valid_prob = calibrated.predict_proba(X_valid)[:, 1]
            test_prob = calibrated.predict_proba(X_test)[:, 1]
            latest_prob = _clip01(calibrated.predict_proba(latest_vector)[0][1])
            probabilities.append(latest_prob)

            base_models[name] = {
                "latest_probability_up": round(latest_prob, 4),
                "latest_vote": "up" if latest_prob >= 0.5 else "down",
                "validation_accuracy": round(float(accuracy_score(y_valid, (valid_prob >= 0.5).astype(int))), 4) if len(y_valid) else 0.0,
                "test_accuracy": round(float(accuracy_score(y_test, (test_prob >= 0.5).astype(int))), 4) if len(y_test) else 0.0,
                "sample_size": int(len(y_test)),
            }

            if name == "gradient_boosting" and hasattr(estimator, "feature_importances_"):
                pairs = list(zip(bundle["features"], estimator.feature_importances_))
                pairs.sort(key=lambda item: item[1], reverse=True)
                feature_importance = [{"feature": feature, "importance": round(float(score), 4)} for feature, score in pairs[:10]]

        ensemble_probability = float(np.mean(probabilities))
        regime, regime_probabilities = self.build_regime_snapshot(df)
        confidence_state = "strong" if len(X_test) >= 30 else "moderate" if len(X_test) >= 15 else "fragile"

        return {
            "base_models": base_models,
            "ensemble_probability_up": round(ensemble_probability, 4),
            "feature_importance": feature_importance,
            "regime_classifier": {"current_regime": regime, "probabilities": regime_probabilities},
            "confidence_governance": {
                "state": confidence_state,
                "minimum_sample_warning": len(df) < 160,
                "degraded_by_regime": regime == "high_volatility",
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
            "volatility": np.tanh((_safe_float(latest.get("Vol_Regime"), 1) - 1) * -2),
            "volume_pressure": np.tanh((_safe_float(latest.get("Relative_Volume"), 1) - 1) * 1.5 + (_safe_float(latest.get("OBV")) / obv_anchor)),
            "downside_risk": np.tanh((_safe_float(latest.get("Downside_Vol_20d"), 0) - _safe_float(latest.get("Realized_Vol_20d"), 0)) * -2 + (_safe_float(latest.get("Drawdown_20d"), 0) * 5)),
        }
        sleeves = {key: round(float(np.clip(value, -1, 1)), 4) for key, value in sleeves.items()}
        model_score = models["ensemble_probability_up"] * 2 - 1
        weighted_signal = np.mean(list(sleeves.values())) if sleeves else 0.0
        composite_index = np.clip(((model_score * 0.5) + (weighted_signal * 0.5) + 1) * 50, 0, 100)

        alpha_rows = []
        for name, value in sleeves.items():
            alpha_rows.append({
                "name": name,
                "score": value,
                "direction": "bullish" if value > 0.15 else "bearish" if value < -0.15 else "neutral",
                "strength": round(abs(value), 4),
                "decay": round(max(0.05, 1 - abs(value) * 0.6), 4),
            })
        alpha_rows.sort(key=lambda item: item["strength"], reverse=True)

        return {
            "composite_index_latest": round(float(composite_index), 2),
            "ensemble_conviction": round(float(composite_index / 100), 4),
            "model_score": round(float(model_score), 4),
            "alpha_sleeves": alpha_rows,
            "confidence_bands": {
                "short_entry": self.config.short_entry,
                "short_exit": self.config.short_exit,
                "long_exit": self.config.long_exit,
                "long_entry": self.config.long_entry,
            },
        }

    def run_validation(self, bundle, execution_mode):
        X = bundle["X"].reset_index(drop=True)
        y = bundle["y"].reset_index(drop=True)
        df = bundle["df"].reset_index(drop=True)
        fold_size = 30
        step = 30 if execution_mode == "quick" else 20
        start = max(120, bundle["splits"]["valid_end"] - 40)

        predictions = []
        weights_history = []
        for test_start in range(start, len(df) - fold_size, step):
            test_end = min(test_start + fold_size, len(df))
            X_train = X.iloc[:test_start]
            y_train = y.iloc[:test_start]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            if len(X_train) < 90 or len(X_test) < 12:
                continue

            local_models = {
                "gb": GradientBoostingClassifier(random_state=42, n_estimators=90 if execution_mode == "quick" else 130, learning_rate=0.05, max_depth=2),
                "rf": RandomForestClassifier(random_state=42, n_estimators=90 if execution_mode == "quick" else 140, max_depth=6, min_samples_leaf=4),
                "lr": LogisticRegression(max_iter=300),
            }

            fold_probs = {}
            model_scores = {}
            for name, estimator in local_models.items():
                estimator.fit(X_train, y_train)
                prob = estimator.predict_proba(X_test)[:, 1]
                fold_probs[name] = prob
                model_scores[name] = accuracy_score(y_test, (prob >= 0.5).astype(int))

            total = sum(model_scores.values()) or 1.0
            weights = {name: score / total for name, score in model_scores.items()}
            weights_history.append(weights)

            for idx, row_idx in enumerate(range(test_start, test_end)):
                ensemble_prob = sum(weights[name] * fold_probs[name][idx] for name in fold_probs)
                predictions.append({
                    "date": df["Date"].iloc[row_idx].strftime("%Y-%m-%d"),
                    "probability_up": float(np.clip(ensemble_prob, 0, 1)),
                    "prediction": int(ensemble_prob >= 0.5),
                    "actual": int(y.iloc[row_idx]),
                })

        if not predictions:
            return {
                "walk_forward": {"anchored_accuracy": 0.0, "rolling_accuracy": 0.0, "folds": 0, "latest_confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}},
                "regime_slices": [],
                "calibration": [],
                "benchmarks": {},
                "weight_stability": {"dominant_model": "n/a", "dispersion": 0.0},
                "warnings": ["Insufficient history for robust walk-forward validation"],
            }

        pred_df = pd.DataFrame(predictions)
        anchored_accuracy = float((pred_df["prediction"] == pred_df["actual"]).mean())
        rolling_window = pred_df.tail(min(60, len(pred_df)))
        rolling_accuracy = float((rolling_window["prediction"] == rolling_window["actual"]).mean())

        pred_df["bucket"] = pd.cut(pred_df["probability_up"], bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0], include_lowest=True)
        calibration = []
        for bucket, group in pred_df.groupby("bucket", observed=False):
            calibration.append({
                "bucket": str(bucket),
                "avg_probability_up": round(float(group["probability_up"].mean()), 4),
                "actual_up_rate": round(float(group["actual"].mean()), 4),
                "count": int(len(group)),
            })

        regime_rows = []
        date_indexed = bundle["df"].set_index(bundle["df"]["Date"].dt.strftime("%Y-%m-%d"))
        for regime_name in ["trending_up", "trending_down", "ranging", "high_volatility"]:
            if regime_name == "trending_up":
                mask = (date_indexed["Trend_60d"] > 0.06) & (date_indexed["ADX"] > 22)
            elif regime_name == "trending_down":
                mask = (date_indexed["Trend_60d"] < -0.06) & (date_indexed["ADX"] > 22)
            elif regime_name == "high_volatility":
                mask = date_indexed["Vol_Regime"] > 1.25
            else:
                mask = ~(((date_indexed["Trend_60d"] > 0.06) & (date_indexed["ADX"] > 22)) | ((date_indexed["Trend_60d"] < -0.06) & (date_indexed["ADX"] > 22)) | (date_indexed["Vol_Regime"] > 1.25))
            regime_dates = set(date_indexed[mask].index.tolist())
            subset = pred_df[pred_df["date"].isin(regime_dates)]
            if len(subset) == 0:
                continue
            regime_rows.append({
                "regime": regime_name,
                "accuracy": round(float((subset["prediction"] == subset["actual"]).mean()), 4),
                "sample_size": int(len(subset)),
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
                "anchored_accuracy": round(anchored_accuracy, 4),
                "rolling_accuracy": round(rolling_accuracy, 4),
                "folds": int(len(weights_history)),
                "latest_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            },
            "regime_slices": regime_rows,
            "calibration": calibration,
            "benchmarks": {
                "buy_hold_hit_rate": round(float(bundle["y"].mean()), 4),
                "naive_momentum_hit_rate": round(float((bundle["df"]["Baseline_Momentum_Signal"]).eq(bundle["y"]).mean()), 4),
            },
            "weight_stability": {
                "dominant_model": dominant_model,
                "dispersion": round(float(np.mean(dispersions) if dispersions else 0.0), 4),
            },
            "warnings": [] if len(bundle["df"]) >= 180 else ["Confidence degraded because history is shorter than preferred"],
        }

    def _candidate_from_signal(self, df, signal_values):
        candidate = df.copy().reset_index(drop=True)
        candidate["RawSignal"] = signal_values
        candidate["Position"] = 0
        candidate.loc[candidate["RawSignal"] >= self.config.long_entry, "Position"] = 1
        candidate.loc[candidate["RawSignal"] <= self.config.short_entry, "Position"] = -1
        candidate["Position"] = candidate["Position"].replace(to_replace=0, method="ffill").fillna(0)
        candidate["Target_Weight"] = np.clip(abs(candidate["RawSignal"] - 0.5) * 2, 0.25, self.config.max_exposure) * candidate["Position"].abs()
        candidate["Market_Return"] = candidate["Close"].pct_change().fillna(0)
        cost = candidate["Position"].shift(1).fillna(0).diff().abs().fillna(0) * ((self.config.slippage_bps + self.config.commission_bps) / 10000)
        candidate["Strategy_Return"] = candidate["Position"].shift(self.config.trade_delay_days).fillna(0) * candidate["Target_Weight"].shift(self.config.trade_delay_days).fillna(0) * candidate["Market_Return"] - cost
        candidate["BuyHold_Value"] = (1 + candidate["Market_Return"]).cumprod()
        candidate["Strategy_Value"] = (1 + candidate["Strategy_Return"]).cumprod()
        candidate["Strategy_Drawdown"] = candidate["Strategy_Value"] / candidate["Strategy_Value"].cummax() - 1
        return candidate

    def _extract_trade_log(self, df):
        trade_log = []
        entry_price = None
        entry_date = None
        entry_side = 0
        for idx in range(1, len(df)):
            prev_pos = int(df["Position"].iloc[idx - 1])
            curr_pos = int(df["Position"].iloc[idx])
            if prev_pos == 0 and curr_pos != 0:
                entry_price = _safe_float(df["Close"].iloc[idx])
                entry_date = df["Date"].iloc[idx].strftime("%Y-%m-%d")
                entry_side = curr_pos
            elif prev_pos != 0 and curr_pos == 0 and entry_price is not None:
                exit_price = _safe_float(df["Close"].iloc[idx])
                gross = ((exit_price / entry_price) - 1) * entry_side
                trade_log.append({
                    "entry_date": entry_date,
                    "exit_date": df["Date"].iloc[idx].strftime("%Y-%m-%d"),
                    "side": "long" if entry_side == 1 else "short",
                    "return_pct": round(gross * 100, 2),
                    "holding_days": int((pd.to_datetime(df["Date"].iloc[idx]) - pd.to_datetime(entry_date)).days),
                })
                entry_price = None
                entry_date = None
                entry_side = 0
        return trade_log

    def _candidate_metrics(self, name, df, regime, rationale, source, policy_thresholds):
        total_return = _safe_float(df["Strategy_Value"].iloc[-1] - 1) * 100
        buy_hold_return = _safe_float(df["BuyHold_Value"].iloc[-1] - 1) * 100
        alpha = total_return - buy_hold_return
        max_drawdown = _safe_float(df["Strategy_Drawdown"].min()) * 100
        turnover = _safe_float(df["Position"].diff().abs().sum())
        stability = max(0.0, 1 - float(df["Strategy_Return"].rolling(20).std().fillna(0).mean()) * 15)
        regime_fit = 1.0 if regime in ["trending_up", "ranging"] else 0.8
        score = (total_return * 0.45) - (abs(max_drawdown) * 0.25) - (turnover * 0.4) + (stability * 10) + (regime_fit * 5)
        latest_position = int(df["Position"].iloc[-1])
        latest_signal = _safe_float(df["RawSignal"].iloc[-1], 0.5)

        if latest_position == 1:
            action = "HOLD_LONG" if latest_signal >= self.config.long_exit else "REDUCE_LONG"
        elif latest_position == -1:
            action = "HOLD_SHORT" if latest_signal <= self.config.short_exit else "REDUCE_SHORT"
        else:
            action = "ENTER_LONG" if latest_signal >= self.config.long_entry else "ENTER_SHORT" if latest_signal <= self.config.short_entry else "WAIT"

        return {
            "name": name,
            "source": source,
            "score": round(score, 2),
            "total_return": round(total_return, 2),
            "buy_hold_return": round(buy_hold_return, 2),
            "alpha": round(alpha, 2),
            "max_drawdown": round(max_drawdown, 2),
            "turnover": round(turnover, 2),
            "stability": round(stability, 4),
            "regime_fit": round(regime_fit, 4),
            "latest_action": action,
            "current_position": latest_position,
            "current_price": round(_safe_float(df["Close"].iloc[-1]), 4),
            "rationale": rationale,
            "thresholds": policy_thresholds,
            "chart_data": {
                "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "strategy": [_safe_float(v) for v in df["Strategy_Value"]],
                "buyhold": [_safe_float(v) for v in df["BuyHold_Value"]],
                "exposure": [_safe_float(v) for v in df["Target_Weight"]],
            },
            "trade_log": self._extract_trade_log(df)[-12:],
        }

    def run_strategy_ranking(self, df, signals, validation, optimize_strategy):
        ranked = []
        regime = validation["weight_stability"]["dominant_model"]

        momentum_signal = np.clip(0.5 + np.tanh(df["Trend_20d"] * 8) * 0.25, 0, 1)
        ranked.append(self._candidate_metrics("Momentum Trend", self._candidate_from_signal(df, momentum_signal), regime, "Trend continuation using Trend_20d", "rule_based", {"long_entry": self.config.long_entry, "short_entry": self.config.short_entry}))

        mean_reversion_signal = np.clip(0.5 + np.tanh((-df["Close_Z20"] * 0.9) + ((50 - df["RSI"]) / 25)) * 0.25, 0, 1)
        ranked.append(self._candidate_metrics("Mean Reversion", self._candidate_from_signal(df, mean_reversion_signal), regime, "Reversal using RSI and close z-score", "rule_based", {"long_entry": self.config.long_entry, "short_entry": self.config.short_entry}))

        breakout_signal = np.clip(0.5 + np.tanh((df["Range_Expansion"] - 1) * 2 + (df["Volume_Regime"] - 1)) * 0.25, 0, 1)
        ranked.append(self._candidate_metrics("Breakout Volume", self._candidate_from_signal(df, breakout_signal), regime, "Range expansion confirmed by volume", "rule_based", {"long_entry": self.config.long_entry, "short_entry": self.config.short_entry}))

        baseline_signal = np.clip(0.35 + df["Baseline_Momentum_Signal"] * 0.3, 0, 1)
        ranked.append(self._candidate_metrics("Baseline Hold Filter", self._candidate_from_signal(df, baseline_signal), regime, "Simple momentum baseline used as fallback floor", "baseline", {"long_entry": self.config.long_entry, "short_entry": self.config.short_entry}))

        meta_details = {"available": False}
        if optimize_strategy:
            try:
                optimizer = StrategyOptimizer(df.copy())
                thresholds = optimizer.optimize_thresholds(min_period=min(180, len(df)))
                if thresholds:
                    meta_df = optimizer.apply_optimal_strategy(thresholds).reset_index(drop=True)
                    meta_df["RawSignal"] = np.clip(meta_df["Composite_Index"] / 100, 0, 1)
                    meta_df["Target_Weight"] = meta_df["Position"].abs() * np.clip(abs(meta_df["RawSignal"] - 0.5) * 2, 0.25, self.config.max_exposure)
                    ranked.append(self._candidate_metrics(
                        "Meta Learner Adaptive",
                        meta_df,
                        thresholds.get("regime", regime),
                        "Adaptive thresholds from the repo meta-learning optimizer",
                        thresholds.get("optimization_method", "meta_learning_ensemble"),
                        {"buy_threshold": round(_safe_float(thresholds.get("buy_threshold", 60)), 2), "sell_threshold": round(_safe_float(thresholds.get("sell_threshold", 40)), 2)},
                    ))
                    meta_details = {
                        "available": True,
                        "thresholds": thresholds,
                        "ensemble_weights": thresholds.get("ensemble_weights", {}),
                        "key_alphas": thresholds.get("key_alphas", {}),
                        "alpha_signals": thresholds.get("alpha_signals", {}),
                        "behavior_analysis": optimizer.get_behavior_analysis(),
                        "alpha_summary": optimizer.get_alpha_summary(),
                        "trader_profiles": optimizer.get_trader_profiles(),
                    }
            except Exception as exc:
                meta_details = {"available": False, "error": str(exc)}

        ranked.sort(key=lambda item: item["score"], reverse=True)
        best = ranked[0]
        baseline = next((item for item in ranked if item["source"] == "baseline"), best)
        promote_baseline = (best["score"] - baseline["score"]) < 2.0 and best["name"] != baseline["name"]
        promoted = baseline if promote_baseline else best
        promotion_reason = "Held baseline because no candidate showed a material edge" if promote_baseline else "Promoted highest-ranked strategy by composite score"

        latest_state = {
            "target_state": "long" if promoted["current_position"] == 1 else "short" if promoted["current_position"] == -1 else "flat",
            "conviction": round(_safe_float(signals["ensemble_conviction"]), 4),
            "position_size_score": round(_safe_float(promoted["chart_data"]["exposure"][-1] if promoted["chart_data"]["exposure"] else 0), 4),
            "expected_horizon_days": 10 if promoted["latest_action"] in ["WAIT", "REDUCE_LONG", "REDUCE_SHORT"] else 20,
            "entry_exit_rationale": promoted["rationale"],
            "recommended_action": promoted["latest_action"],
            "current_price": promoted["current_price"],
            "current_position": promoted["current_position"],
            "strategy_thresholds": promoted["thresholds"],
            "promoted_strategy": promoted["name"],
            "risk_flags": [
                "Fallback to baseline" if promote_baseline else "Promoted strategy has measurable edge",
                "Confidence degraded by sample size" if validation["warnings"] else "Validation window acceptable",
            ],
        }

        performance = {
            "strategy_total_return": promoted["total_return"],
            "buy_hold_total_return": promoted["buy_hold_return"],
            "alpha_vs_buy_hold": promoted["alpha"],
            "max_drawdown": promoted["max_drawdown"],
            "tail_loss_5pct": round(_safe_float(df["Close"].pct_change().fillna(0).quantile(0.05)) * 100, 2),
            "turnover": promoted["turnover"],
            "gross_exposure": round(float(np.mean(promoted["chart_data"]["exposure"])) if promoted["chart_data"]["exposure"] else 0.0, 4),
            "avg_holding_days": round(float(np.mean([trade["holding_days"] for trade in promoted["trade_log"]])) if promoted["trade_log"] else 0.0, 2),
            "trade_count": len(promoted["trade_log"]),
            "win_rate": round((len([t for t in promoted["trade_log"] if t["return_pct"] > 0]) / len(promoted["trade_log"]) * 100) if promoted["trade_log"] else 0.0, 2),
            "expectancy": round(float(np.mean([trade["return_pct"] for trade in promoted["trade_log"]])) if promoted["trade_log"] else 0.0, 2),
            "payoff_ratio": 0.0,
            "stability": promoted["stability"],
            "regime_fit": promoted["regime_fit"],
        }

        return {
            "candidates": ranked[:5],
            "promoted_policy": {
                "name": promoted["name"],
                "source": promoted["source"],
                "score": promoted["score"],
                "promotion_reason": promotion_reason,
                "material_edge_over_baseline": round(best["score"] - baseline["score"], 2),
            },
            "latest_state": latest_state,
            "performance": performance,
            "trade_log": promoted["trade_log"],
            "chart_data": promoted["chart_data"],
            "meta_learner": meta_details,
            "warnings": validation["warnings"] + ([] if meta_details.get("available", False) else ["Meta learner unavailable or not promoted"]),
        }

    def build_charts(self, df, signals, strategy):
        tail = df.tail(120).copy()
        strategy_tail = strategy["chart_data"]
        strategy_map = {date: idx for idx, date in enumerate(strategy_tail["dates"])}

        tail["Composite_Index"] = np.linspace(max(0, signals["composite_index_latest"] - 12), signals["composite_index_latest"], len(tail))
        tail["Strategy_Value"] = [strategy_tail["strategy"][strategy_map.get(date.strftime("%Y-%m-%d"), 0)] for date in tail["Date"]]
        tail["BuyHold_Value"] = [strategy_tail["buyhold"][strategy_map.get(date.strftime("%Y-%m-%d"), 0)] for date in tail["Date"]]
        tail["Target_Weight"] = [strategy_tail["exposure"][strategy_map.get(date.strftime("%Y-%m-%d"), 0)] for date in tail["Date"]]

        comparison = strategy["candidates"][1] if len(strategy["candidates"]) > 1 else strategy["candidates"][0]

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
                "comparison_name": comparison["name"],
                "comparison_strategy": [round((_safe_float(v) - 1) * 100, 4) for v in comparison["chart_data"]["strategy"][-len(tail):]],
            },
            "alpha_heatmap": signals["alpha_sleeves"],
        }

    def build_summary(self, symbol, period, df, data_quality, signals, validation, strategy):
        promoted = strategy["promoted_policy"]
        latest = strategy["latest_state"]
        recommendation = f"{promoted['name']} -> {latest['recommended_action']}"
        return {
            "symbol": symbol,
            "period": period,
            "last_close": round(_safe_float(df["Close"].iloc[-1]), 4),
            "day_change_pct": round(_safe_float(df["Close"].pct_change().iloc[-1]) * 100, 2),
            "recommendation": recommendation,
            "target_state": latest["target_state"],
            "recommended_action": latest["recommended_action"],
            "current_regime": validation["weight_stability"]["dominant_model"],
            "ensemble_conviction": round(_safe_float(signals["ensemble_conviction"]), 4),
            "composite_index": signals["composite_index_latest"],
            "research_score": round(promoted["score"] + (data_quality["score"] * 0.15) + (validation["walk_forward"]["rolling_accuracy"] * 20), 2),
            "promoted_strategy": promoted["name"],
        }

    def build_explanations(self, summary, signals, validation, strategy, data_quality):
        active = signals["alpha_sleeves"][:3]
        active_text = ", ".join([f"{item['name']} ({item['direction']})" for item in active]) if active else "no dominant sleeves"
        return {
            "overview": f"{summary['symbol']} promotes {summary['promoted_strategy']} with action {summary['recommended_action']}. Composite conviction is {signals['composite_index_latest']:.1f}/100.",
            "why_now": f"Top alpha sleeves are {active_text}. Walk-forward rolling accuracy is {validation['walk_forward']['rolling_accuracy']:.2%}.",
            "risk": f"Data quality is {data_quality['quality_state']}. Promotion warnings: {'; '.join(strategy['warnings']) if strategy['warnings'] else 'none'}.",
            "execution": f"Current price is {strategy['latest_state']['current_price']} with current position {strategy['latest_state']['current_position']}. Thresholds: {strategy['latest_state']['strategy_thresholds']}.",
        }


def run_quant_research(symbol, period="2y", optimize_strategy=True, execution_mode="quick"):
    workstation = QuantResearchWorkstation()
    result = workstation.run(symbol=symbol, period=period, optimize_strategy=optimize_strategy, execution_mode=execution_mode)
    return json.dumps(result)


async def run_quant_research_async(symbol, period="2y", optimize_strategy=True, execution_mode="quick"):
    workstation = QuantResearchWorkstation()
    df_raw = await get_stock_data_async(ticker=symbol, period=period)
    result = workstation.run_with_raw_data(symbol=symbol, period=period, df_raw=df_raw, optimize_strategy=optimize_strategy, execution_mode=execution_mode)
    return json.dumps(result)
