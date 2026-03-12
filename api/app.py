from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import get_stock_data
from feature_engineering import add_technical_indicators, prepare_data_for_model
from model import StockPredictor
from strategy_optimizer import StrategyOptimizer
from factor_analysis import (
    get_fundamental_data, 
    calculate_factor_score, 
    add_factors_to_df,
    get_factor_comparison,
    format_fundamentals_for_display
)
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        period = data.get('period', '2y')
        optimize = data.get('optimize', True)

        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400

        # Fetch stock data
        print(f"Fetching data for {symbol}...")
        df = get_stock_data(symbol, period)

        # Add technical indicators
        print(f"Adding technical indicators...")
        df_features = add_technical_indicators(df)

        # Prepare data for model
        print(f"Preparing data for model...")
        X_train, y_train, latest_features, recent_data = prepare_data_for_model(df_features)

        # Train/load model
        print(f"Training/loading model...")
        predictor = StockPredictor()
        model_path = f"{symbol}_model.pkl"

        if os.path.exists(model_path):
            predictor.load_model(model_path)
            predictor.calculate_recent_accuracy(recent_data)
        else:
            predictor.train_model(X_train, y_train, recent_data)
            predictor.save_model(model_path)

        # Make prediction
        prediction, probability = predictor.predict(latest_features)

        # Calculate composite index
        print(f"Calculating composite index...")
        df_features_copy = df_features.copy()
        composite_index = predictor.calculate_composite_index(df_features_copy)
        df_features_copy['Composite_Index'] = composite_index

        # Optimize strategy using smart meta-learning optimizer
        strategy_thresholds = {'buy_threshold': 60, 'sell_threshold': 40}
        strategy_metrics = None
        optimization_info = None

        if optimize:
            print(f"Running smart meta-learning optimization with alpha factors...")
            optimizer = StrategyOptimizer(df_features_copy)
            optimal_thresholds = optimizer.optimize_thresholds(min_period=120)

            if optimal_thresholds:
                df_features_copy = optimizer.apply_optimal_strategy(optimal_thresholds)
                strategy_thresholds = optimal_thresholds

                # Get behavior analysis and alpha summary
                behavior_analysis = optimizer.get_behavior_analysis()
                alpha_summary = optimizer.get_alpha_summary()

                # Extract optimization info for frontend display
                optimization_info = {
                    'method': optimal_thresholds.get('optimization_method', 'grid_search'),
                    'regime': optimal_thresholds.get('regime', 'unknown'),
                    'ensemble_weights': optimal_thresholds.get('ensemble_weights', {}),
                    'win_rate': optimal_thresholds.get('win_rate', 0),
                    'num_trades': optimal_thresholds.get('num_trades', 0),
                    'profit_factor': optimal_thresholds.get('profit_factor', 0),
                    'key_alphas': optimal_thresholds.get('key_alphas', {}),
                    'alpha_signals': optimal_thresholds.get('alpha_signals', {})
                }

                # Extract behavior analysis summary
                behavior_summary = {}
                if behavior_analysis:
                    if 'strategy_profile' in behavior_analysis:
                        sp = behavior_analysis['strategy_profile']
                        behavior_summary['style'] = sp.get('style', 'Unknown')
                        behavior_summary['selectivity'] = sp.get('selectivity', 'Unknown')
                        behavior_summary['long_exposure'] = sp.get('long_exposure', 0)
                        behavior_summary['short_exposure'] = sp.get('short_exposure', 0)
                        behavior_summary['cash_exposure'] = sp.get('cash_exposure', 0)

                    if 'indicator_usage' in behavior_analysis:
                        iu = behavior_analysis['indicator_usage']
                        behavior_summary['primary_indicators'] = [s['indicator'] for s in iu.get('primary_signals', [])]
                        behavior_summary['confirmation_indicators'] = [s['indicator'] for s in iu.get('confirmation_signals', [])]

                    if 'trading_summary' in behavior_analysis:
                        behavior_summary['summary'] = behavior_analysis['trading_summary']

                    if 'risk_profile' in behavior_analysis:
                        rp = behavior_analysis['risk_profile']
                        behavior_summary['risk_metrics'] = {
                            'avg_daily_return': rp.get('avg_daily_return', 0),
                            'max_drawdown': rp.get('max_drawdown', 0),
                            'positive_days_pct': rp.get('positive_days_pct', 0)
                        }

                # Add alpha interpretations
                alpha_interpretations = {}
                if alpha_summary and 'message' not in alpha_summary:
                    for category, data in alpha_summary.items():
                        if isinstance(data, dict) and 'interpretation' in data:
                            alpha_interpretations[category] = data['interpretation']

                # Calculate strategy metrics
                last_120 = df_features_copy.iloc[-120:]
                strategy_metrics = {
                    'period': f"{len(last_120)} days",
                    'buyhold_return': ((last_120['BuyHold_Value'].iloc[-1] / last_120['BuyHold_Value'].iloc[0] - 1) * 100),
                    'strategy_return': ((last_120['Strategy_Value'].iloc[-1] / last_120['Strategy_Value'].iloc[0] - 1) * 100),
                    'strategy_max_dd': (last_120['Strategy_Drawdown'].min() * 100),
                    'strategy_sharpe': optimal_thresholds.get('sharpe_ratio', 0),
                    'alpha': optimal_thresholds.get('total_return', 0),
                    'beta': 1.0,
                    'trades': int(last_120['Position'].diff().fillna(0).abs().sum() / 2),
                    'optimization_info': optimization_info,
                    'behavior_summary': behavior_summary,
                    'alpha_interpretations': alpha_interpretations
                }

        df_features = df_features_copy

        # Prepare response
        response = {
            'symbol': symbol,
            'prediction': int(prediction),
            'probability': float(probability),
            'metrics': {
                'accuracy': predictor.recent_metrics['accuracy'],
                'f1_score': predictor.recent_metrics['f1_score'],
                'up_precision': predictor.recent_metrics['up_precision'],
                'down_precision': predictor.recent_metrics['down_precision'],
                'up_count': predictor.recent_metrics['up_count'],
                'down_count': predictor.recent_metrics['down_count'],
                'correct_predictions': predictor.recent_metrics['correct_predictions'],
                'total_predictions': predictor.recent_metrics['total_predictions'],
                'strategy_metrics': strategy_metrics
            },
            'price_data': {
                'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist()[-200:],
                'close': df['Close'].tolist()[-200:],
                'ma20': df['MA20'].tolist()[-200:] if 'MA20' in df.columns else [],
                'ma50': df['MA50'].tolist()[-200:] if 'MA50' in df.columns else []
            },
            'composite_data': {
                'dates': df_features['Date'].dt.strftime('%Y-%m-%d').tolist()[-200:],
                'values': df_features['Composite_Index'].tolist()[-200:]
            },
            'thresholds': {
                'buy_threshold': strategy_thresholds.get('buy_threshold', 60),
                'sell_threshold': strategy_thresholds.get('sell_threshold', 40)
            },
            'strategy_data': {
                'dates': [],
                'buyhold': [],
                'strategy': []
            }
        }

        # Add strategy performance data
        if 'BuyHold_Value' in df_features.columns and 'Strategy_Value' in df_features.columns:
            recent_df = df_features.iloc[-120:]
            initial_bh = recent_df['BuyHold_Value'].iloc[0]
            initial_st = recent_df['Strategy_Value'].iloc[0]

            response['strategy_data'] = {
                'dates': recent_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'buyhold': ((recent_df['BuyHold_Value'] / initial_bh - 1) * 100).tolist(),
                'strategy': ((recent_df['Strategy_Value'] / initial_st - 1) * 100).tolist()
            }

        return jsonify(response)

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e)}), 500


@app.route('/api/factors', methods=['POST'])
def analyze_factors():
    """
    Analyze fundamental factors for a stock.
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()

        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400

        # Fetch fundamental data
        print(f"Fetching fundamental data for {symbol}...")
        fundamentals = get_fundamental_data(symbol)
        
        # Format for display
        formatted = format_fundamentals_for_display(fundamentals)
        
        # Calculate factor scores
        scores = calculate_factor_score(fundamentals)
        
        return jsonify({
            'symbol': symbol,
            'fundamentals': formatted,
            'factor_scores': scores
        })

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/factors/compare', methods=['POST'])
def compare_factors():
    """
    Compare factors across multiple stocks.
    """
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])

        if not tickers:
            return jsonify({'error': 'At least one ticker is required'}), 400

        # Get comparison data
        print(f"Comparing factors for {tickers}...")
        comparison_df = get_factor_comparison(tickers)
        
        # Convert to dict for JSON
        comparison = comparison_df.to_dict('records')
        
        return jsonify({
            'tickers': tickers,
            'comparison': comparison
        })

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/factors/compare-full', methods=['POST'])
def compare_factors_full():
    """
    Full factor analysis with technical indicators combined.
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        period = data.get('period', '2y')

        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400

        # Fetch both price and fundamental data
        print(f"Fetching data for {symbol}...")
        df = get_stock_data(symbol, period)
        
        print(f"Fetching fundamental data for {symbol}...")
        fundamentals = get_fundamental_data(symbol)
        
        # Add technical indicators
        print(f"Adding technical indicators...")
        df_features = add_technical_indicators(df)
        
        # Add fundamental factors
        print(f"Adding fundamental factors...")
        df_factors = add_factors_to_df(df_features, fundamentals)
        
        # Get factor scores
        scores = calculate_factor_score(fundamentals)
        
        # Format fundamentals
        formatted = format_fundamentals_for_display(fundamentals)
        
        return jsonify({
            'symbol': symbol,
            'fundamentals': formatted,
            'factor_scores': scores,
            'technical_indicators': {
                'rsi': df_factors['RSI'].iloc[-1] if 'RSI' in df_factors.columns else None,
                'macd': df_factors['MACD'].iloc[-1] if 'MACD' in df_factors.columns else None,
                'macd_signal': df_factors['MACD_signal'].iloc[-1] if 'MACD_signal' in df_factors.columns else None,
                'macd_hist': df_factors['MACD_hist'].iloc[-1] if 'MACD_hist' in df_factors.columns else None,
                'bb_upper': df_factors['BB_upper'].iloc[-1] if 'BB_upper' in df_factors.columns else None,
                'bb_lower': df_factors['BB_lower'].iloc[-1] if 'BB_lower' in df_factors.columns else None,
                'adx': df_factors['ADX'].iloc[-1] if 'ADX' in df_factors.columns else None,
                'price': df_factors['Close'].iloc[-1],
                'ma20': df_factors['MA20'].iloc[-1] if 'MA20' in df_factors.columns else None,
                'ma50': df_factors['MA50'].iloc[-1] if 'MA50' in df_factors.columns else None,
            },
            'combined_signals': {
                'technical': 'bullish' if df_factors['RSI'].iloc[-1] < 30 else ('bearish' if df_factors['RSI'].iloc[-1] > 70 else 'neutral') if 'RSI' in df_factors.columns else 'unknown',
                'value': 'undervalued' if scores['value_score'] > 60 else ('overvalued' if scores['value_score'] < 40 else 'fair'),
                'growth': 'high' if scores['growth_score'] > 60 else ('low' if scores['growth_score'] < 40 else 'average'),
                'profitability': 'strong' if scores['profitability_score'] > 60 else ('weak' if scores['profitability_score'] < 40 else 'average'),
                'overall': 'buy' if scores['overall_score'] > 65 else ('sell' if scores['overall_score'] < 35 else 'hold')
            }
        })

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Stock Predictor API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
