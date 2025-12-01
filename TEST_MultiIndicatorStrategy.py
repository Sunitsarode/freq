"""
Simple test script to verify strategy loads and indicators calculate correctly
Run from freqtrade root directory: python test_strategy.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add strategies path
sys.path.insert(0, str(Path('user_data/strategies')))
sys.path.insert(0, str(Path('user_data/strategies/sumit_ma')))

# Import strategy
from MultiIndicatorStrategy import MultiIndicatorStrategy

def test_strategy():
    print("="*60)
    print("TESTING MULTI-INDICATOR STRATEGY")
    print("="*60)
    
    # Create minimal config for strategy
    config = {
        'stake_currency': 'USDT',
        'dry_run': True,
        'exchange': {
            'name': 'binance',
            'pair_whitelist': ['BTC/USDT:USDT'],
        }
    }
    
    # Create strategy instance with config
    strategy = MultiIndicatorStrategy(config)
    print(f"\n✓ Strategy loaded: {strategy.__class__.__name__}")
    print(f"  Timeframe: {strategy.timeframe}")
    print(f"  Can Short: {strategy.can_short}")
    print(f"  Startup Candles: {strategy.startup_candle_count}")
    
    # Create sample dataframe
    print("\n📊 Creating test dataframe...")
    dates = pd.date_range(start='2024-01-01', periods=500, freq='5min')
    
    df = pd.DataFrame({
        'date': dates,
        'open': 50000 + (pd.Series(range(500)) * 10),
        'high': 50100 + (pd.Series(range(500)) * 10),
        'low': 49900 + (pd.Series(range(500)) * 10),
        'close': 50000 + (pd.Series(range(500)) * 10),
        'volume': 1000000
    })
    
    df = df.set_index('date')
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Test populate_indicators
    print("\n🔧 Testing populate_indicators...")
    try:
        df = strategy.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
        print("  ✓ Indicators calculated successfully")
        
        # Check which indicators were added
        indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        print(f"  Added {len(indicator_cols)} indicator columns:")
        for col in indicator_cols[:10]:  # Show first 10
            print(f"    - {col}")
        if len(indicator_cols) > 10:
            print(f"    ... and {len(indicator_cols) - 10} more")
        
        # Check for NaN values
        nan_counts = df[indicator_cols].isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if len(cols_with_nan) > 0:
            print(f"\n  ⚠ Columns with NaN values:")
            for col, count in cols_with_nan.items():
                print(f"    - {col}: {count} NaNs")
        else:
            print(f"\n  ✓ No NaN values in indicators")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test populate_entry_trend
    print("\n📈 Testing populate_entry_trend...")
    try:
        df = strategy.populate_entry_trend(df, {'pair': 'BTC/USDT:USDT'})
        print("  ✓ Entry signals calculated")
        
        long_entries = df['enter_long'].sum()
        short_entries = df['enter_short'].sum() if 'enter_short' in df.columns else 0
        
        print(f"  Long entries: {long_entries}")
        print(f"  Short entries: {short_entries}")
        
    except Exception as e:
        print(f"  ✗ Error in populate_entry_trend: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test populate_exit_trend
    print("\n📉 Testing populate_exit_trend...")
    try:
        df = strategy.populate_exit_trend(df, {'pair': 'BTC/USDT:USDT'})
        print("  ✓ Exit signals calculated")
        
        long_exits = df['exit_long'].sum()
        short_exits = df['exit_short'].sum() if 'exit_short' in df.columns else 0
        
        print(f"  Long exits: {long_exits}")
        print(f"  Short exits: {short_exits}")
        
    except Exception as e:
        print(f"  ✗ Error in populate_exit_trend: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check data types
    print("\n🔍 Checking data types...")
    print(f"  enter_long dtype: {df['enter_long'].dtype}")
    print(f"  exit_long dtype: {df['exit_long'].dtype}")
    
    # Check for datetime columns
    datetime_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns, UTC]' or 'datetime' in str(df[col].dtype)]
    if datetime_cols:
        print(f"\n  ⚠ Found datetime columns: {datetime_cols}")
    else:
        print(f"\n  ✓ No datetime columns in dataframe")
    
    # Sample data
    print("\n📋 Sample data (last 5 rows):")
    sample_cols = ['close', 'rsi_5m', 'adx_5m', 'enter_long', 'exit_long']
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].tail())
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_strategy()
    sys.exit(0 if success else 1)
