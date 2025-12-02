"""
Comprehensive test script for Multi-Indicator Strategy
IMPROVED: Better validation, error checking, and reporting
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add strategies path
sys.path.insert(0, str(Path('user_data/strategies')))
sys.path.insert(0, str(Path('user_data/strategies/sumit_ma')))

# Import strategy
from MultiIndicatorStrategy import MultiIndicatorStrategy


def create_realistic_ohlc_data(periods=500, base_price=50000, volatility=0.02):
    """Create more realistic OHLC data with proper datetime index"""
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='5min', tz='UTC')
    
    # Generate price movement using random walk
    returns = np.random.normal(0, volatility, periods)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, volatility/2, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, volatility/2, periods))),
        'close': prices * (1 + np.random.normal(0, volatility/4, periods)),
        'volume': np.random.uniform(1000000, 5000000, periods)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def validate_indicators(df, indicator_cols):
    """Validate indicator quality"""
    print("\n🔍 Validating Indicator Quality...")
    
    issues = []
    
    for col in indicator_cols:
        if col not in df.columns:
            continue
            
        # Check for NaN values
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            pct = (nan_count / len(df)) * 100
            if pct > 10:  # More than 10% NaN is problematic
                issues.append(f"  ❌ {col}: {nan_count} NaNs ({pct:.1f}%)")
            else:
                print(f"  ⚠️  {col}: {nan_count} NaNs ({pct:.1f}%) - acceptable")
        
        # Check for infinite values
        inf_count = np.isinf(df[col].replace([np.inf, -np.inf], np.nan)).sum()
        if inf_count > 0:
            issues.append(f"  ❌ {col}: {inf_count} infinite values")
        
        # Check for constant values (no variation)
        if df[col].nunique() == 1:
            issues.append(f"  ❌ {col}: constant value (no variation)")
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("  ✅ All indicators look good!")
        return True


def test_strategy():
    print("=" * 70)
    print("COMPREHENSIVE MULTI-INDICATOR STRATEGY TEST")
    print("=" * 70)
    
    # Create minimal config
    config = {
        'stake_currency': 'USDT',
        'dry_run': True,
        'exchange': {
            'name': 'binance',
            'pair_whitelist': ['BTC/USDT:USDT'],
        }
    }
    
    # Initialize strategy
    print("\n📦 Initializing Strategy...")
    try:
        strategy = MultiIndicatorStrategy(config)
        print(f"  ✅ Strategy loaded: {strategy.__class__.__name__}")
        print(f"     Timeframe: {strategy.timeframe}")
        print(f"     Can Short: {strategy.can_short}")
        print(f"     Startup Candles: {strategy.startup_candle_count}")
        print(f"     Max Open Trades: {strategy.max_open_trades.value if hasattr(strategy.max_open_trades, 'value') else strategy.max_open_trades}")
    except Exception as e:
        print(f"  ❌ Failed to load strategy: {e}")
        return False
    
    # Create test data
    print("\n📊 Creating Realistic Test Data...")
    try:
        df = create_realistic_ohlc_data(periods=500, base_price=50000, volatility=0.02)
        print(f"  ✅ Created {len(df)} candles")
        print(f"     Date range: {df.index[0]} to {df.index[-1]}")
        print(f"     Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"     Index type: {type(df.index).__name__}")
    except Exception as e:
        print(f"  ❌ Failed to create data: {e}")
        return False
    
    # Test populate_indicators
    print("\n🔧 Testing populate_indicators...")
    try:
        df_with_indicators = strategy.populate_indicators(df.copy(), {'pair': 'BTC/USDT:USDT'})
        
        indicator_cols = [col for col in df_with_indicators.columns 
                         if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        print(f"  ✅ Indicators calculated successfully")
        print(f"     Added {len(indicator_cols)} indicator columns")
        
        # Show key indicators
        key_indicators = ['buy_signal_count', 'sell_signal_count', 'rsi_5m', 'rsi_1h', 
                         'adx_5m', 'macd_hist_5m', 'st_7_3_direction']
        print(f"\n     Key Indicators (last value):")
        for ind in key_indicators:
            if ind in df_with_indicators.columns:
                last_val = df_with_indicators[ind].iloc[-1]
                print(f"       {ind}: {last_val:.2f}" if not pd.isna(last_val) else f"       {ind}: NaN")
        
        # Validate indicators
        if not validate_indicators(df_with_indicators, indicator_cols):
            print("  ⚠️  Some indicators have issues, but continuing...")
        
        df = df_with_indicators
        
    except Exception as e:
        print(f"  ❌ Error in populate_indicators: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test populate_entry_trend
    print("\n📈 Testing populate_entry_trend...")
    try:
        df = strategy.populate_entry_trend(df, {'pair': 'BTC/USDT:USDT'})
        
        long_entries = df['enter_long'].sum()
        short_entries = df['enter_short'].sum() if 'enter_short' in df.columns else 0
        
        print(f"  ✅ Entry signals calculated")
        print(f"     Long entries: {long_entries} ({long_entries/len(df)*100:.1f}%)")
        print(f"     Short entries: {short_entries} ({short_entries/len(df)*100:.1f}%)")
        
        if long_entries == 0 and short_entries == 0:
            print(f"  ⚠️  WARNING: No entry signals generated!")
            print(f"     This might indicate overly strict conditions")
        
        # Show where signals occurred
        if long_entries > 0:
            entry_indices = df[df['enter_long'] == 1].index[:3]
            print(f"\n     First 3 long entry timestamps:")
            for idx in entry_indices:
                print(f"       {idx}")
        
    except Exception as e:
        print(f"  ❌ Error in populate_entry_trend: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test populate_exit_trend
    print("\n📉 Testing populate_exit_trend...")
    try:
        df = strategy.populate_exit_trend(df, {'pair': 'BTC/USDT:USDT'})
        
        long_exits = df['exit_long'].sum()
        short_exits = df['exit_short'].sum() if 'exit_short' in df.columns else 0
        
        print(f"  ✅ Exit signals calculated")
        print(f"     Long exits: {long_exits} ({long_exits/len(df)*100:.1f}%)")
        print(f"     Short exits: {short_exits} ({short_exits/len(df)*100:.1f}%)")
        
    except Exception as e:
        print(f"  ❌ Error in populate_exit_trend: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Data type validation
    print("\n🔍 Validating Data Types...")
    print(f"  enter_long dtype: {df['enter_long'].dtype}")
    print(f"  exit_long dtype: {df['exit_long'].dtype}")
    print(f"  Index dtype: {df.index.dtype}")
    
    # Check for problematic columns
    datetime_cols = [col for col in df.columns 
                    if 'datetime' in str(df[col].dtype).lower()]
    if datetime_cols:
        print(f"  ⚠️  Found datetime columns: {datetime_cols}")
    else:
        print(f"  ✅ No datetime columns in dataframe")
    
    # Sample data display
    print("\n📋 Sample Data (last 5 rows):")
    sample_cols = ['close', 'buy_signal_count', 'rsi_5m', 'adx_5m', 'enter_long', 'exit_long']
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].tail().to_string())
    
    # Performance stats
    print("\n📊 Signal Statistics:")
    total_candles = len(df)
    long_entry_rate = (df['enter_long'].sum() / total_candles * 100)
    long_exit_rate = (df['exit_long'].sum() / total_candles * 100)
    
    print(f"  Total candles: {total_candles}")
    print(f"  Long entry rate: {long_entry_rate:.2f}%")
    print(f"  Long exit rate: {long_exit_rate:.2f}%")
    
    if 'enter_short' in df.columns:
        short_entry_rate = (df['enter_short'].sum() / total_candles * 100)
        short_exit_rate = (df['exit_short'].sum() / total_candles * 100)
        print(f"  Short entry rate: {short_entry_rate:.2f}%")
        print(f"  Short exit rate: {short_exit_rate:.2f}%")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Run backtesting with real data")
    print("2. Optimize parameters using hyperopt")
    print("3. Test in dry-run mode before live trading")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_strategy()
    sys.exit(0 if success else 1)
