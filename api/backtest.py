"""
Backtesting Engine for Stock Trading Strategies

A lightweight, vectorized backtesting library designed for browser (Pyodide) compatibility.
Built on top of pandas/numpy for efficient computation.

Features:
- Vectorized backtesting for speed
- Multiple entry/exit signal types
- Comprehensive performance metrics
- Trade-level analysis
- Equity curve and drawdown calculations
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class BacktestResult:
    """Container for backtest results."""
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.positions = []
        self.signals = []
        
    def to_dict(self):
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'positions': self.positions,
            'signals': self.signals,
            'summary': self.summary
        }


class Backtester:
    """
    Vectorized Backtesting Engine
    
    Supports:
    - Long/Short/Long-Short strategies
    - Multiple entry/exit signal types
    - Configurable commission and slippage
    - Walk-forward testing
    """
    
    def __init__(self, 
                 initial_capital=10000,
                 commission=0.001,
                 slippage=0.0005,
                 position_sizing='equal'):
        """
        Initialize backtester.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        commission : float
            Commission rate (0.001 = 0.1%)
        slippage : float
            Slippage rate
        position_sizing : str
            'equal' or 'fixed'
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing
        
    def run(self, 
            df, 
            entry_signal, 
            exit_signal=None,
            direction='long',
            price_col='Close',
            verbose=True):
        """
        Run backtest.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with at least Close column
        entry_signal : pd.Series
            Boolean series, True = enter position
        exit_signal : pd.Series
            Boolean series, True = exit position
        direction : str
            'long', 'short', or 'both'
        price_col : str
            Column name for prices
            
        Returns:
        --------
        dict : Backtest results and metrics
        """
        df = df.copy()
        n = len(df)
        
        # Ensure signals are aligned
        if isinstance(entry_signal, list):
            entry_signal = pd.Series(entry_signal, index=df.index)
        if exit_signal is not None and isinstance(exit_signal, list):
            exit_signal = pd.Series(exit_signal, index=df.index)
            
        # Initialize position series
        position = pd.Series(0, index=df.index)
        
        # Track position state
        in_position = False
        position_type = 0  # 1 for long, -1 for short
        entry_price = 0
        entry_idx = 0
        
        trades = []
        
        for i in range(1, n):
            prev_position = position.iloc[i-1]
            
            if not in_position:
                # Check for entry signal
                if entry_signal.iloc[i]:
                    # Apply slippage for entry
                    if direction in ['long', 'both']:
                        entry_price = df[price_col].iloc[i] * (1 + self.slippage)
                        position_type = 1
                    else:
                        entry_price = df[price_col].iloc[i] * (1 - self.slippage)
                        position_type = -1
                    
                    in_position = True
                    entry_idx = i
                    
            else:
                # Check for exit signal
                should_exit = False
                
                if exit_signal is not None and exit_signal.iloc[i]:
                    should_exit = True
                    
                # Also exit if opposite signal (for both direction)
                if direction == 'both' and entry_signal.iloc[i]:
                    should_exit = True
                    
                if should_exit:
                    # Apply slippage for exit
                    if position_type == 1:
                        exit_price = df[price_col].iloc[i] * (1 - self.slippage)
                    else:
                        exit_price = df[price_col].iloc[i] * (1 + self.slippage)
                    
                    # Calculate trade return
                    if position_type == 1:
                        ret = (exit_price - entry_price) / entry_price
                    else:
                        ret = (entry_price - exit_price) / entry_price
                    
                    # Subtract commission
                    ret -= self.commission * 2
                    
                    trades.append({
                        'entry_date': df.index[entry_idx],
                        'exit_date': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': ret,
                        'position_type': 'long' if position_type == 1 else 'short',
                        'holding_days': i - entry_idx
                    })
                    
                    in_position = False
                    position_type = 0
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, df, position, price_col)
        
        if verbose:
            self._print_results(metrics, trades)
            
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': self._get_equity_curve(trades, df),
            'positions': position.tolist()
        }
    
    def run_bracket(self,
                    df,
                    entry_signal,
                    take_profit=None,
                    stop_loss=None,
                    trailing_stop=None,
                    time_exit=None,
                    direction='long',
                    price_col='Close',
                    verbose=True):
        """
        Run backtest with bracket orders (TP/SL).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data
        entry_signal : pd.Series
            Entry signal (boolean)
        take_profit : float
            Take profit level (e.g., 0.05 = 5%)
        stop_loss : float
            Stop loss level (e.g., 0.03 = 3%)
        trailing_stop : float
            Trailing stop percentage
        time_exit : int
            Exit after N bars
        direction : str
            'long' or 'short'
        """
        df = df.copy()
        n = len(df)
        
        if isinstance(entry_signal, list):
            entry_signal = pd.Series(entry_signal, index=df.index)
            
        position = pd.Series(0, index=df.index)
        trades = []
        
        in_position = False
        entry_price = 0
        entry_idx = 0
        stop_price = 0
        tp_price = 0
        
        for i in range(1, n):
            if not in_position:
                if entry_signal.iloc[i]:
                    # Enter position
                    if direction == 'long':
                        entry_price = df[price_col].iloc[i] * (1 + self.slippage)
                        if stop_loss:
                            stop_price = entry_price * (1 - stop_loss)
                        if take_profit:
                            tp_price = entry_price * (1 + take_profit)
                    else:
                        entry_price = df[price_col].iloc[i] * (1 - self.slippage)
                        if stop_loss:
                            stop_price = entry_price * (1 + stop_loss)
                        if take_profit:
                            tp_price = entry_price * (1 - take_profit)
                    
                    in_position = True
                    entry_idx = i
                    position.iloc[i] = 1 if direction == 'long' else -1
                    
            else:
                current_price = df[price_col].iloc[i]
                high_price = df.get('High', df[price_col]).iloc[i]
                low_price = df.get('Low', df[price_col]).iloc[i]
                
                should_exit = False
                exit_reason = ''
                
                # Check time exit
                if time_exit and (i - entry_idx) >= time_exit:
                    should_exit = True
                    exit_reason = 'time'
                
                # Check stop loss / take profit
                if direction == 'long':
                    if stop_loss and low_price <= stop_price:
                        should_exit = True
                        exit_reason = 'stop_loss'
                        exit_price = stop_price
                    elif take_profit and high_price >= tp_price:
                        should_exit = True
                        exit_reason = 'take_profit'
                        exit_price = tp_price
                else:
                    if stop_loss and high_price >= stop_price:
                        should_exit = True
                        exit_reason = 'stop_loss'
                        exit_price = stop_price
                    elif take_profit and low_price <= tp_price:
                        should_exit = True
                        exit_reason = 'take_profit'
                        exit_price = tp_price
                
                # Update trailing stop
                if trailing_stop and direction == 'long':
                    new_stop = current_price * (1 - trailing_stop)
                    if new_stop > stop_price:
                        stop_price = new_stop
                elif trailing_stop and direction == 'short':
                    new_stop = current_price * (1 + trailing_stop)
                    if new_stop < stop_price:
                        stop_price = new_stop
                
                if should_exit:
                    if exit_reason in ['stop_loss', 'take_profit']:
                        exit_price = exit_price * (1 - self.slippage) if direction == 'long' else exit_price * (1 + self.slippage)
                    else:
                        exit_price = current_price * (1 - self.slippage) if direction == 'long' else current_price * (1 + self.slippage)
                    
                    if direction == 'long':
                        ret = (exit_price - entry_price) / entry_price
                    else:
                        ret = (entry_price - exit_price) / entry_price
                    
                    ret -= self.commission * 2
                    
                    trades.append({
                        'entry_date': df.index[entry_idx],
                        'exit_date': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': ret,
                        'exit_reason': exit_reason,
                        'holding_days': i - entry_idx
                    })
                    
                    in_position = False
                    position.iloc[i] = 0
                else:
                    position.iloc[i] = position.iloc[i-1]
        
        metrics = self._calculate_metrics(trades, df, position, price_col)
        
        if verbose:
            self._print_results(metrics, trades, bracket=True)
            
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': self._get_equity_curve(trades, df),
            'positions': position.tolist()
        }
    
    def run_indicator_based(self,
                           df,
                           indicator_col,
                           entry_above=None,
                           entry_below=None,
                           exit_above=None,
                           exit_below=None,
                           direction='long',
                           price_col='Close',
                           verbose=True):
        """
        Run backtest based on indicator values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with indicator column
        indicator_col : str
            Column name for indicator
        entry_above : float
            Enter when indicator crosses above this level
        entry_below : float
            Enter when indicator crosses below this level
        exit_above : float
            Exit when indicator crosses above this level
        exit_below : float
            Exit when indicator crosses below this level
        direction : str
            'long' or 'short'
        """
        indicator = df[indicator_col]
        
        # Generate entry/exit signals
        entry_signal = pd.Series(False, index=df.index)
        exit_signal = pd.Series(False, index=df.index)
        
        if entry_above is not None:
            entry_signal = (indicator > entry_above) & (indicator.shift(1) <= entry_above)
        if entry_below is not None:
            entry_signal = entry_signal | ((indicator < entry_below) & (indicator.shift(1) >= entry_below))
            
        if exit_above is not None:
            exit_signal = (indicator > exit_above) & (indicator.shift(1) <= exit_above)
        if exit_below is not None:
            exit_signal = exit_signal | ((indicator < exit_below) & (indicator.shift(1) >= exit_below))
        
        return self.run(df, entry_signal, exit_signal, direction, price_col, verbose)
    
    def _calculate_metrics(self, trades, df, positions, price_col):
        """Calculate comprehensive performance metrics."""
        
        if not trades:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'avg_holding_days': 0,
                'max_consecutive_losses': 0
            }
        
        returns = [t['return'] for t in trades]
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r <= 0]
        
        # Basic metrics
        total_return = sum(returns)
        total_trades = len(trades)
        wins = len(winning_returns)
        losses = len(losing_returns)
        
        win_rate = wins / total_trades if total_trades > 0 else 0
        profit_factor = abs(sum(winning_returns) / sum(losing_returns)) if sum(losing_returns) != 0 else 0
        
        # Average metrics
        avg_win = np.mean(winning_returns) if winning_returns else 0
        avg_loss = np.mean(losing_returns) if losing_returns else 0
        avg_holding = np.mean([t['holding_days'] for t in trades])
        
        # Calculate equity curve
        equity = [self.initial_capital]
        for ret in returns:
            equity.append(equity[-1] * (1 + ret))
        
        # Drawdown
        peak = equity[0]
        max_dd = 0
        dd_list = []
        
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            dd_list.append(dd)
            if dd > max_dd:
                max_dd = dd
        
        # Time metrics
        total_days = (df.index[-1] - df.index[0]).days if hasattr(df.index[-1], 'days') else len(df)
        years = total_days / 365
        
        # Annualized return
        if years > 0:
            annualized_return = ((1 + total_return) ** (1 / years) - 1) if total_return > -1 else -1
        else:
            annualized_return = 0
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        if returns and np.std(returns) > 0:
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe = 0
        
        # Sortino Ratio (downside deviation)
        if returns:
            downside_returns = [r for r in returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 1
            if downside_std > 0:
                sortino = np.sqrt(252) * np.mean(returns) / downside_std
            else:
                sortino = 0
        else:
            sortino = 0
        
        # Calmar Ratio
        if max_dd > 0:
            calmar = annualized_return / max_dd
        else:
            calmar = 0
        
        # Max consecutive losses
        max_consecutive = 0
        current_consecutive = 0
        for r in returns:
            if r < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return {
            'total_return': total_return * 100,  # As percentage
            'annualized_return': annualized_return * 100,
            'max_drawdown': max_dd * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': losses,
            'avg_win': avg_win * 100,
            'avg_loss': avg_loss * 100,
            'avg_holding_days': avg_holding,
            'max_consecutive_losses': max_consecutive,
            'final_equity': equity[-1] if equity else self.initial_capital
        }
    
    def _get_equity_curve(self, trades, df):
        """Generate equity curve data."""
        equity = [self.initial_capital]
        dates = [df.index[0]]
        
        trade_idx = 0
        current_value = self.initial_capital
        
        for i in range(len(df)):
            if trade_idx < len(trades) and df.index[i] >= trades[trade_idx]['exit_date']:
                current_value *= (1 + trades[trade_idx]['return'])
                trade_idx += 1
            
            equity.append(current_value)
            dates.append(df.index[i])
        
        return {
            'dates': [str(d)[:10] for d in dates],
            'values': equity
        }
    
    def _print_results(self, metrics, trades, bracket=False):
        """Print backtest results."""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return:     {metrics['total_return']:.2f}%")
        print(f"Annualized:       {metrics['annualized_return']:.2f}%")
        print(f"Max Drawdown:     {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio:    {metrics['sortino_ratio']:.3f}")
        print(f"Calmar Ratio:     {metrics['calmar_ratio']:.3f}")
        print("-"*50)
        print(f"Total Trades:     {metrics['total_trades']}")
        print(f"Win Rate:         {metrics['win_rate']:.1f}%")
        print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
        print(f"Avg Win:          {metrics['avg_win']:.2f}%")
        print(f"Avg Loss:         {metrics['avg_loss']:.2f}%")
        print(f"Avg Holding:      {metrics['avg_holding_days']:.1f} days")
        print("="*50)


def compare_strategies(df, strategies, initial_capital=10000):
    """
    Compare multiple strategies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    strategies : dict
        Dict of {name: backtest_result}
    initial_capital : float
        Starting capital
        
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    results = []
    
    for name, result in strategies.items():
        metrics = result['metrics']
        results.append({
            'Strategy': name,
            'Return (%)': metrics['total_return'],
            'Ann. Return (%)': metrics['annualized_return'],
            'Max DD (%)': metrics['max_drawdown'],
            'Sharpe': metrics['sharpe_ratio'],
            'Win Rate (%)': metrics['win_rate'],
            'Trades': metrics['total_trades'],
            'Profit Factor': metrics['profit_factor']
        })
    
    return pd.DataFrame(results).sort_values('Return (%)', ascending=False)


def walk_forward_backtest(backtester, df, params, train_window=252, test_window=63, **backtest_kwargs):
    """
    Perform walk-forward backtesting.
    
    Parameters:
    -----------
    backtester : Backtester instance
    df : pd.DataFrame
        Full price data
    params : dict
        Strategy parameters to test
    train_window : int
        Training period length
    test_window : int
        Testing period length
    **backtest_kwargs
        Arguments passed to backtester.run()
        
    Returns:
    --------
    dict : Walk-forward results
    """
    results = []
    
    i = train_window
    while i + test_window <= len(df):
        train_df = df.iloc[i-train_window:i]
        test_df = df.iloc[i:i+test_window]
        
        # Optimize on training data (simplified - just use params)
        # In practice, you'd run optimization here
        
        # Test on test data
        test_result = backtester.run(test_df, **backtest_kwargs)
        
        results.append({
            'train_start': train_df.index[0],
            'train_end': train_df.index[-1],
            'test_start': test_df.index[0],
            'test_end': test_df.index[-1],
            'return': test_result['metrics']['total_return'],
            'sharpe': test_result['metrics']['sharpe_ratio']
        })
        
        i += test_window
    
    return results


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    
    df = pd.DataFrame({
        'Close': prices,
        'High': prices * (1 + np.abs(np.random.randn(n) * 0.01)),
        'Low': prices * (1 - np.abs(np.random.randn(n) * 0.01))
    }, index=dates)
    
    # Add some indicators
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = 50 + np.random.randn(n) * 20  # Fake RSI
    
    # Run backtest
    bt = Backtester(initial_capital=10000)
    
    # Signal: Enter when price crosses above MA20
    entry = (df['Close'] > df['MA20']) & (df['Close'].shift(1) <= df['MA20'].shift(1))
    
    # Exit when price crosses below MA20 or RSI > 70
    exit_signal = (df['Close'] < df['MA20']) & (df['Close'].shift(1) >= df['MA20'].shift(1))
    
    result = bt.run(df, entry, exit_signal)
    
    print("\nSimple MA Crossover Backtest:")
    print(f"Return: {result['metrics']['total_return']:.2f}%")
    print(f"Sharpe: {result['metrics']['sharpe_ratio']:.3f}")
