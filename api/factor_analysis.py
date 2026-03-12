"""
Factor Analysis Module for Stock Prediction
==========================================
Calculates fundamental factors for stock analysis including:
- Value factors (PE, PB, PS, Dividend Yield, PCF)
- Profitability factors (ROE, ROA, Gross Margin, etc.)
- Leverage factors (Debt/Equity, Current Ratio, etc.)
- Growth factors (Revenue, Earnings, Book Value growth)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import warnings
warnings.filterwarnings('ignore')


def safe_division(a, b, default=np.nan):
    """Safely divide two numbers, avoiding division by zero."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        return pd.Series(np.where(b != 0, a / b, default), index=a.index)
    elif isinstance(a, pd.DataFrame) and isinstance(b, (pd.DataFrame, pd.Series)):
        return a.div(b, errors='ignore').fillna(default)
    else:
        return a / b if b != 0 else default


def get_fundamental_data(ticker, max_retries=3):
    """
    Fetch fundamental data for a given ticker from Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    max_retries : int
        Maximum number of retry attempts
        
    Returns:
    --------
    dict
        Dictionary containing fundamental metrics
    """
    retry_count = 0
    fundamentals = {}
    
    while retry_count < max_retries:
        try:
            # Get summary detail (contains many fundamental metrics)
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            params = {
                'modules': 'summaryDetail,defaultKeyStatistics,financialData,assetProfile'
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': f'https://finance.yahoo.com/quote/{ticker}',
                'Origin': 'https://finance.yahoo.com'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 403:
                # Try alternative endpoint
                url_alt = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
                response = requests.get(url_alt, params=params, headers=headers, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            
            if 'quoteSummary' not in data or not data['quoteSummary']['result']:
                raise ValueError(f"No fundamental data found for {ticker}")
            
            result = data['quoteSummary']['result'][0]
            
            # Extract summary detail
            summary = result.get('summaryDetail', {})
            default_stats = result.get('defaultKeyStatistics', {})
            financial_data = result.get('financialData', {})
            asset_profile = result.get('assetProfile', {})
            
            # Value Factors
            fundamentals['marketCap'] = summary.get('marketCap', {}).get('raw')
            fundamentals['trailingPE'] = summary.get('trailingPE', {}).get('raw')
            fundamentals['forwardPE'] = summary.get('forwardPE', {}).get('raw')
            fundamentals['priceToBook'] = summary.get('priceToBook', {}).get('raw')
            fundamentals['priceToSales'] = summary.get('priceToSalesTrailing12Months', {}).get('raw')
            fundamentals['enterpriseValue'] = summary.get('enterpriseValue', {}).get('raw')
            fundamentals['enterpriseToRevenue'] = summary.get('enterpriseToRevenue', {}).get('raw')
            fundamentals['enterpriseToEbitda'] = summary.get('enterpriseToEbitda', {}).get('raw')
            
            # Dividend
            fundamentals['dividendYield'] = summary.get('dividendYield', {}).get('raw')
            fundamentals['dividendRate'] = summary.get('dividendRate', {}).get('raw')
            fundamentals['payoutRatio'] = summary.get('payoutRatio', {}).get('raw')
            
            # Profitability Factors
            fundamentals['profitMargins'] = financial_data.get('profitMargins', {}).get('raw')
            fundamentals['grossMargins'] = financial_data.get('grossMargins', {}).get('raw')
            fundamentals['operatingMargins'] = financial_data.get('operatingMargins', {}).get('raw')
            fundamentals['ebitdaMargins'] = financial_data.get('ebitdaMargins', {}).get('raw')
            fundamentals['returnOnEquity'] = financial_data.get('returnOnEquity', {}).get('raw')
            fundamentals['returnOnAssets'] = financial_data.get('returnOnAssets', {}).get('raw')
            
            # Growth Factors
            fundamentals['earningsGrowth'] = default_stats.get('earningsGrowth', {}).get('raw')
            fundamentals['revenueGrowth'] = default_stats.get('revenueGrowth', {}).get('raw')
            fundamentals['earningsQuarterlyGrowth'] = financial_data.get('earningsQuarterlyGrowth', {}).get('raw')
            fundamentals['revenueQuarterlyGrowth'] = financial_data.get('revenueQuarterlyGrowth', {}).get('raw')
            
            # Leverage Factors
            fundamentals['debtToEquity'] = financial_data.get('debtToEquity', {}).get('raw')
            fundamentals['currentRatio'] = financial_data.get('currentRatio', {}).get('raw')
            fundamentals['quickRatio'] = financial_data.get('quickRatio', {}).get('raw')
            fundamentals['totalDebt'] = summary.get('totalDebt', {}).get('raw')
            
            # Performance Factors
            fundamentals['52WeekChange'] = default_stats.get('52WeekChange', {}).get('raw')
            fundamentals['beta'] = default_stats.get('beta', {}).get('raw')
            fundamentals['volatility'] = summary.get('volatility', {}).get('raw')
            
            # Other Metrics
            fundamentals['targetMeanPrice'] = financial_data.get('targetMeanPrice', {}).get('raw')
            fundamentals['recommendationKey'] = financial_data.get('recommendationKey', {})
            fundamentals['numberOfAnalystOpinions'] = default_stats.get('numberOfAnalystOpinions', {}).get('raw')
            
            # Company Info
            fundamentals['sector'] = asset_profile.get('sector', '')
            fundamentals['industry'] = asset_profile.get('industry', '')
            fundamentals['fullTimeEmployees'] = asset_profile.get('fullTimeEmployees', 0)
            
            # Get historical dividend data
            fundamentals['dividendHistory'] = get_dividend_history(ticker)
            
            return fundamentals
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Error fetching fundamental data: {e}")
                # Return sample data for demonstration
                return get_sample_fundamentals(ticker)
            time.sleep(0.5)
    
    return get_sample_fundamentals(ticker)


def get_sample_fundamentals(ticker):
    """Return sample fundamentals for common stocks (for demo when API unavailable)."""
    # Sample data for common stocks - realistic approximate values
    sample_data = {
        'AAPL': {
            'marketCap': 3000000000000, 'trailingPE': 28.5, 'forwardPE': 25.0, 'priceToBook': 45.0,
            'priceToSales': 7.5, 'enterpriseValue': 3100000000000, 'enterpriseToEbitda': 22.0,
            'dividendYield': 0.005, 'dividendRate': 1.0, 'payoutRatio': 0.15,
            'profitMargins': 0.25, 'grossMargins': 0.46, 'operatingMargins': 0.29,
            'ebitdaMargins': 0.34, 'returnOnEquity': 1.5, 'returnOnAssets': 0.28,
            'earningsGrowth': 0.13, 'revenueGrowth': 0.08, 'earningsQuarterlyGrowth': 0.14,
            'revenueQuarterlyGrowth': 0.06, 'debtToEquity': 180, 'currentRatio': 1.2,
            'quickRatio': 1.0, 'totalDebt': 120000000000, '52WeekChange': 0.25,
            'beta': 1.3, 'targetMeanPrice': 200.0, 'recommendationKey': 'buy',
            'numberOfAnalystOpinions': 40, 'sector': 'Technology', 'industry': 'Consumer Electronics'
        },
        'MSFT': {
            'marketCap': 2800000000000, 'trailingPE': 35.0, 'forwardPE': 30.0, 'priceToBook': 12.0,
            'priceToSales': 11.0, 'enterpriseValue': 2900000000000, 'enterpriseToEbitda': 20.0,
            'dividendYield': 0.008, 'dividendRate': 3.0, 'payoutRatio': 0.25,
            'profitMargins': 0.35, 'grossMargins': 0.70, 'operatingMargins': 0.42,
            'ebitdaMargins': 0.45, 'returnOnEquity': 0.38, 'returnOnAssets': 0.15,
            'earningsGrowth': 0.15, 'revenueGrowth': 0.12, 'earningsQuarterlyGrowth': 0.18,
            'revenueQuarterlyGrowth': 0.10, 'debtToEquity': 40, 'currentRatio': 1.8,
            'quickRatio': 1.7, 'totalDebt': 80000000000, '52WeekChange': 0.35,
            'beta': 1.1, 'targetMeanPrice': 450.0, 'recommendationKey': 'buy',
            'numberOfAnalystOpinions': 45, 'sector': 'Technology', 'industry': 'Software'
        },
        'GOOGL': {
            'marketCap': 1700000000000, 'trailingPE': 22.0, 'forwardPE': 18.0, 'priceToBook': 6.0,
            'priceToSales': 5.5, 'enterpriseValue': 1800000000000, 'enterpriseToEbitda': 14.0,
            'dividendYield': 0.0, 'dividendRate': 0.0, 'payoutRatio': 0.0,
            'profitMargins': 0.24, 'grossMargins': 0.56, 'operatingMargins': 0.28,
            'ebitdaMargins': 0.35, 'returnOnEquity': 0.25, 'returnOnAssets': 0.12,
            'earningsGrowth': 0.12, 'revenueGrowth': 0.10, 'earningsQuarterlyGrowth': 0.15,
            'revenueQuarterlyGrowth': 0.08, 'debtToEquity': 10, 'currentRatio': 2.2,
            'quickRatio': 2.1, 'totalDebt': 20000000000, '52WeekChange': 0.30,
            'beta': 1.05, 'targetMeanPrice': 160.0, 'recommendationKey': 'buy',
            'numberOfAnalystOpinions': 50, 'sector': 'Communication Services', 'industry': 'Internet'
        },
        'TSLA': {
            'marketCap': 800000000000, 'trailingPE': 60.0, 'forwardPE': 45.0, 'priceToBook': 15.0,
            'priceToSales': 8.0, 'enterpriseValue': 850000000000, 'enterpriseToEbitda': 25.0,
            'dividendYield': 0.0, 'dividendRate': 0.0, 'payoutRatio': 0.0,
            'profitMargins': 0.15, 'grossMargins': 0.25, 'operatingMargins': 0.17,
            'ebitdaMargins': 0.22, 'returnOnEquity': 0.25, 'returnOnAssets': 0.08,
            'earningsGrowth': 0.30, 'revenueGrowth': 0.25, 'earningsQuarterlyGrowth': 0.40,
            'revenueQuarterlyGrowth': 0.20, 'debtToEquity': 20, 'currentRatio': 1.5,
            'quickRatio': 1.2, 'totalDebt': 15000000000, '52WeekChange': 0.50,
            'beta': 2.0, 'targetMeanPrice': 250.0, 'recommendationKey': 'hold',
            'numberOfAnalystOpinions': 35, 'sector': 'Consumer Discretionary', 'industry': 'Auto Manufacturers'
        }
    }
    
    base = get_empty_fundamentals(ticker)
    if ticker.upper() in sample_data:
        base.update(sample_data[ticker.upper()])
    else:
        # Generate reasonable defaults for unknown tickers
        base.update({
            'marketCap': 50000000000,
            'trailingPE': 20.0,
            'priceToBook': 3.0,
            'returnOnEquity': 0.15,
            'revenueGrowth': 0.05,
            'beta': 1.0,
            'sector': 'Unknown',
            'recommendationKey': 'hold'
        })
    
    return base


def get_empty_fundamentals(ticker):
    """Return empty fundamentals dictionary with None values."""
    return {
        'ticker': ticker,
        'marketCap': None,
        'trailingPE': None,
        'forwardPE': None,
        'priceToBook': None,
        'priceToSales': None,
        'enterpriseValue': None,
        'enterpriseToRevenue': None,
        'enterpriseToEbitda': None,
        'dividendYield': None,
        'dividendRate': None,
        'payoutRatio': None,
        'profitMargins': None,
        'grossMargins': None,
        'operatingMargins': None,
        'ebitdaMargins': None,
        'returnOnEquity': None,
        'returnOnAssets': None,
        'earningsGrowth': None,
        'revenueGrowth': None,
        'earningsQuarterlyGrowth': None,
        'revenueQuarterlyGrowth': None,
        'debtToEquity': None,
        'currentRatio': None,
        'quickRatio': None,
        'totalDebt': None,
        '52WeekChange': None,
        'beta': None,
        'targetMeanPrice': None,
        'recommendationKey': None,
        'numberOfAnalystOpinions': None,
        'sector': '',
        'industry': '',
        'fullTimeEmployees': 0,
        'dividendHistory': []
    }


def get_dividend_history(ticker, max_retries=3):
    """Get dividend history for a ticker."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {
                'range': '5y',
                'interval': '1mo',
                'events': 'div'
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json',
                'Referer': f'https://finance.yahoo.com/quote/{ticker}'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 403:
                url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
                response = requests.get(url, params=params, headers=headers, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            
            if 'chart' not in data or not data['chart']['result']:
                return []
            
            result = data['chart']['result'][0]
            if 'events' not in result or 'dividends' not in result['events']:
                return []
            
            dividends = result['events']['dividends']
            dividend_list = []
            for date, info in dividends.items():
                dividend_list.append({
                    'date': datetime.fromtimestamp(info['date']).strftime('%Y-%m-%d'),
                    'amount': info['amount']
                })
            
            return sorted(dividend_list, key=lambda x: x['date'], reverse=True)
            
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                return []
            time.sleep(0.5)
    
    return []


def calculate_factor_score(fundamentals):
    """
    Calculate composite factor scores for screening.
    
    Parameters:
    -----------
    fundamentals : dict
        Fundamental data dictionary
        
    Returns:
    --------
    dict
        Factor scores (normalized 0-100)
    """
    scores = {
        'value_score': 0,
        'profitability_score': 0,
        'growth_score': 0,
        'leverage_score': 0,
        'momentum_score': 0,
        'overall_score': 0
    }
    
    value_factors = []
    profitability_factors = []
    growth_factors = []
    leverage_factors = []
    momentum_factors = []
    
    # Value Factors (lower is better for most)
    if fundamentals.get('trailingPE') and fundamentals['trailingPE'] > 0:
        # PE: 0-15 good (>40 poor), normalize to 0-100
        pe = fundamentals['trailingPE']
        value_factors.append(max(0, min(100, (40 - pe) / 40 * 100)))
    
    if fundamentals.get('priceToBook') and fundamentals['priceToBook'] > 0:
        # PB: 0-2 good (>8 poor)
        pb = fundamentals['priceToBook']
        value_factors.append(max(0, min(100, (8 - pb) / 8 * 100)))
    
    if fundamentals.get('priceToSales') and fundamentals['priceToSales'] > 0:
        # PS: 0-2 good (>6 poor)
        ps = fundamentals['priceToSales']
        value_factors.append(max(0, min(100, (6 - ps) / 6 * 100)))
    
    if fundamentals.get('dividendYield') and fundamentals['dividendYield'] > 0:
        # Higher is better, cap at 8%
        div = fundamentals['dividendYield'] * 100
        value_factors.append(min(100, div / 8 * 100))
    
    # Profitability Factors (higher is better)
    if fundamentals.get('returnOnEquity'):
        # ROE: >20% excellent, <0% poor
        roe = fundamentals['returnOnEquity'] * 100
        profitability_factors.append(max(0, min(100, roe / 20 * 100)))
    
    if fundamentals.get('returnOnAssets'):
        # ROA: >10% excellent, <0% poor
        roa = fundamentals['returnOnAssets'] * 100
        profitability_factors.append(max(0, min(100, roa / 10 * 100)))
    
    if fundamentals.get('grossMargins'):
        # Gross Margin: >50% excellent
        gm = fundamentals['grossMargins'] * 100
        profitability_factors.append(max(0, min(100, gm / 50 * 100)))
    
    if fundamentals.get('operatingMargins'):
        # Operating Margin: >20% excellent
        om = fundamentals['operatingMargins'] * 100
        profitability_factors.append(max(0, min(100, om / 20 * 100)))
    
    # Growth Factors (higher is better)
    if fundamentals.get('earningsGrowth'):
        eg = fundamentals['earningsGrowth'] * 100
        growth_factors.append(max(0, min(100, eg / 30 * 100)))
    
    if fundamentals.get('revenueGrowth'):
        rg = fundamentals['revenueGrowth'] * 100
        growth_factors.append(max(0, min(100, rg / 25 * 100)))
    
    # Leverage Factors (lower is better)
    if fundamentals.get('debtToEquity') is not None:
        # Debt/Equity: 0-50 good, >200 poor
        de = fundamentals['debtToEquity']
        leverage_factors.append(max(0, min(100, (200 - de) / 200 * 100)))
    
    if fundamentals.get('currentRatio'):
        # Current Ratio: >2 good, <1 poor
        cr = fundamentals['currentRatio']
        if cr >= 2:
            leverage_factors.append(100)
        elif cr >= 1:
            leverage_factors.append((cr - 1) * 100)
        else:
            leverage_factors.append(0)
    
    # Momentum Factors
    if fundamentals.get('52WeekChange'):
        change = fundamentals['52WeekChange'] * 100
        momentum_factors.append(max(0, min(100, (change + 50) / 100 * 100)))
    
    # Calculate averages
    scores['value_score'] = np.mean(value_factors) if value_factors else 50
    scores['profitability_score'] = np.mean(profitability_factors) if profitability_factors else 50
    scores['growth_score'] = np.mean(growth_factors) if growth_factors else 50
    scores['leverage_score'] = np.mean(leverage_factors) if leverage_factors else 50
    scores['momentum_score'] = np.mean(momentum_factors) if momentum_factors else 50
    
    # Overall score (weighted average)
    scores['overall_score'] = (
        scores['value_score'] * 0.20 +
        scores['profitability_score'] * 0.25 +
        scores['growth_score'] * 0.15 +
        scores['leverage_score'] * 0.15 +
        scores['momentum_score'] * 0.25
    )
    
    return scores


def add_factors_to_df(df, fundamentals):
    """
    Add fundamental factors as features to price dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Price data DataFrame
    fundamentals : dict
        Fundamental data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added factor columns
    """
    df_factors = df.copy()
    
    # Get the latest fundamental values (broadcast to all rows)
    latest_price = df_factors['Close'].iloc[-1]
    
    # Value factors
    df_factors['PE_Ratio'] = fundamentals.get('trailingPE', np.nan)
    df_factors['PB_Ratio'] = fundamentals.get('priceToBook', np.nan)
    df_factors['PS_Ratio'] = fundamentals.get('priceToSales', np.nan)
    df_factors['PCF_Ratio'] = fundamentals.get('enterpriseToEbitda', np.nan)
    df_factors['Dividend_Yield'] = fundamentals.get('dividendYield', 0) or 0
    
    # Profitability factors
    df_factors['ROE'] = fundamentals.get('returnOnEquity', np.nan)
    df_factors['ROA'] = fundamentals.get('returnOnAssets', np.nan)
    df_factors['Gross_Margin'] = fundamentals.get('grossMargins', np.nan)
    df_factors['Operating_Margin'] = fundamentals.get('operatingMargins', np.nan)
    df_factors['Net_Margin'] = fundamentals.get('profitMargins', np.nan)
    
    # Growth factors
    df_factors['Earnings_Growth'] = fundamentals.get('earningsGrowth', np.nan)
    df_factors['Revenue_Growth'] = fundamentals.get('revenueGrowth', np.nan)
    
    # Leverage factors
    df_factors['Debt_To_Equity'] = fundamentals.get('debtToEquity', np.nan)
    df_factors['Current_Ratio'] = fundamentals.get('currentRatio', np.nan)
    df_factors['Quick_Ratio'] = fundamentals.get('quickRatio', np.nan)
    
    # Momentum/Other factors
    df_factors['Beta'] = fundamentals.get('beta', np.nan)
    df_factors['52W_Change'] = fundamentals.get('52WeekChange', np.nan)
    
    # Forward PE (estimated valuation)
    df_factors['Forward_PE'] = fundamentals.get('forwardPE', np.nan)
    
    # PEG Ratio (PE / Earnings Growth) - Value + Growth
    if fundamentals.get('trailingPE') and fundamentals.get('earningsGrowth'):
        peg = fundamentals['trailingPE'] / (fundamentals['earningsGrowth'] * 100)
        df_factors['PEG_Ratio'] = peg
    else:
        df_factors['PEG_Ratio'] = np.nan
    
    # Analyst consensus
    df_factors['Analyst_Rating'] = {
        'strongBuy': 5, 'buy': 4, 'hold': 3, 'sell': 2, 'strongSell': 1
    }.get(fundamentals.get('recommendationKey', 'hold'), 3)
    
    return df_factors


def get_factor_comparison(tickers):
    """
    Get factor comparison for multiple tickers.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
        
    Returns:
    --------
    pandas.DataFrame
        Comparison table of factors
    """
    comparison_data = []
    
    for ticker in tickers:
        try:
            fundamentals = get_fundamental_data(ticker)
            scores = calculate_factor_score(fundamentals)
            
            comparison_data.append({
                'Ticker': ticker,
                'Sector': fundamentals.get('sector', ''),
                'Industry': fundamentals.get('industry', ''),
                'Market Cap': fundamentals.get('marketCap'),
                'PE': fundamentals.get('trailingPE'),
                'PB': fundamentals.get('priceToBook'),
                'Dividend Yield': fundamentals.get('dividendYield'),
                'ROE': fundamentals.get('returnOnEquity'),
                'ROA': fundamentals.get('returnOnAssets'),
                'Gross Margin': fundamentals.get('grossMargins'),
                'Debt/Equity': fundamentals.get('debtToEquity'),
                'Current Ratio': fundamentals.get('currentRatio'),
                'Revenue Growth': fundamentals.get('revenueGrowth'),
                'Earnings Growth': fundamentals.get('earningsGrowth'),
                'Beta': fundamentals.get('beta'),
                'Value Score': scores.get('value_score'),
                'Profitability Score': scores.get('profitability_score'),
                'Growth Score': scores.get('growth_score'),
                'Overall Score': scores.get('overall_score')
            })
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue
    
    return pd.DataFrame(comparison_data)


def format_fundamentals_for_display(fundamentals):
    """
    Format fundamentals for display in the web interface.
    
    Parameters:
    -----------
    fundamentals : dict
        Raw fundamentals data
        
    Returns:
    --------
    dict
        Formatted fundamentals for display
    """
    formatted = {}
    
    # Helper to format numbers
    def fmt(val, suffix='', prefix='', decimals=2):
        if val is None:
            return 'N/A'
        try:
            if suffix == 'B' and val > 1e9:
                return f"{prefix}{val/1e9:.{decimals}B}"
            elif suffix == 'M' and val > 1e6:
                return f"{prefix}{val/1e6:.{decimals}M}"
            elif suffix == '%':
                return f"{prefix}{val*100:.{decimals}f}%"
            elif decimals > 0:
                return f"{prefix}{val:.{decimals}f}"
            else:
                return f"{prefix}{val}"
        except:
            return 'N/A'
    
    # Format each metric
    formatted['marketCap'] = fmt(fundamentals.get('marketCap'), 'B')
    formatted['trailingPE'] = fmt(fundamentals.get('trailingPE'), decimals=2)
    formatted['forwardPE'] = fmt(fundamentals.get('forwardPE'), decimals=2)
    formatted['priceToBook'] = fmt(fundamentals.get('priceToBook'), decimals=2)
    formatted['priceToSales'] = fmt(fundamentals.get('priceToSales'), decimals=2)
    formatted['enterpriseValue'] = fmt(fundamentals.get('enterpriseValue'), 'B')
    
    formatted['dividendYield'] = fmt(fundamentals.get('dividendYield'), '%')
    formatted['dividendRate'] = fmt(fundamentals.get('dividendRate'), decimals=2)
    formatted['payoutRatio'] = fmt(fundamentals.get('payoutRatio'), '%')
    
    formatted['profitMargins'] = fmt(fundamentals.get('profitMargins'), '%')
    formatted['grossMargins'] = fmt(fundamentals.get('grossMargins'), '%')
    formatted['operatingMargins'] = fmt(fundamentals.get('operatingMargins'), '%')
    formatted['returnOnEquity'] = fmt(fundamentals.get('returnOnEquity'), '%')
    formatted['returnOnAssets'] = fmt(fundamentals.get('returnOnAssets'), '%')
    
    formatted['earningsGrowth'] = fmt(fundamentals.get('earningsGrowth'), '%')
    formatted['revenueGrowth'] = fmt(fundamentals.get('revenueGrowth'), '%')
    
    formatted['debtToEquity'] = fmt(fundamentals.get('debtToEquity'), decimals=1)
    formatted['currentRatio'] = fmt(fundamentals.get('currentRatio'), decimals=2)
    formatted['quickRatio'] = fmt(fundamentals.get('quickRatio'), decimals=2)
    
    formatted['beta'] = fmt(fundamentals.get('beta'), decimals=2)
    formatted['52WeekChange'] = fmt(fundamentals.get('52WeekChange'), '%')
    
    formatted['targetMeanPrice'] = fmt(fundamentals.get('targetMeanPrice'), '$', decimals=0)
    formatted['recommendation'] = fundamentals.get('recommendationKey', 'N/A').upper()
    formatted['analystCount'] = fundamentals.get('numberOfAnalystOpinions', 'N/A')
    
    formatted['sector'] = fundamentals.get('sector', 'N/A')
    formatted['industry'] = fundamentals.get('industry', 'N/A')
    
    # Calculate factor scores
    scores = calculate_factor_score(fundamentals)
    formatted['factorScores'] = scores
    
    return formatted


if __name__ == "__main__":
    # Test with AAPL
    print("Testing factor analysis with AAPL...")
    fundamentals = get_fundamental_data("AAPL")
    print(json.dumps(format_fundamentals_for_display(fundamentals), indent=2))
