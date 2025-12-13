# -*- coding: utf-8 -*-
"""
Custom Indicators Module for Freqtrade Strategy
Contains all indicator calculations for multi-timeframe analysis
FIXED: Proper resampling compatible with Freqtrade's dataframe structure
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict
import logging

logger = logging.getLogger(__name__)
def sumit_ma_signals(df: pd.DataFrame, sma1_period: int = 21, sma2_period: int = 51, 
                     normalize_window: int = 100) -> pd.DataFrame:
    """
    SUMIT MA-AVG SIGNAL LOGIC (Indicator Only)
    
    Calculations:
    1. MA3, MA51, MA201, MA301 on OHLC/4
    2. indicator_value1 = MA3 - MA301
    3. indicator_value2 = MA201 - MA301
    4. indicator_sma1 = SMA(indicator_value1, 21)
    5. indicator_sma2 = SMA(indicator_value1, 51)
    6. signal_count normalized to 0-100 range (like RSI)
    
    Returns all calculated values for strategy to use
    """
    
    # Calculate typical price (OHLC/4)
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Calculate MAs
    ma3 = talib.SMA(ohlc4, timeperiod=3)
    ma51 = talib.SMA(ohlc4, timeperiod=51)
    ma201 = talib.SMA(ohlc4, timeperiod=201)
    ma301 = talib.SMA(ohlc4, timeperiod=301)
    
    # Calculate indicator values
    indicator_value1 = ma3 - ma301
    indicator_value2 = ma201 - ma301
    
    # Calculate SMAs of indicator_value1
    indicator_sma1 = talib.SMA(indicator_value1, timeperiod=sma1_period)
    indicator_sma2 = talib.SMA(indicator_value1, timeperiod=sma2_period)
    
    # Normalize indicator_value1 to 0-100 range (like RSI)
    rolling_min = indicator_value1.rolling(window=normalize_window, min_periods=1).min()
    rolling_max = indicator_value1.rolling(window=normalize_window, min_periods=1).max()
    
    # Avoid division by zero
    range_diff = rolling_max - rolling_min
    range_diff = range_diff.replace(0, 1)  # Replace 0 with 1 to avoid division by zero
    
    # Normalize to 0-100
    signal_count_normalized = ((indicator_value1 - rolling_min) / range_diff) * 100
    signal_count_normalized = signal_count_normalized.fillna(50)  # Fill NaN with neutral 50
    signal_count_normalized = signal_count_normalized.clip(0, 100)  # Ensure 0-100 range
    
    # Build output DataFrame
    result = pd.DataFrame(index=df.index)
    
    # Main indicator values (keeping naming for strategy compatibility)
    result['signal_count'] = signal_count_normalized  # Normalized to 0-100
    result['signal_ma3'] = indicator_sma1             # SMA(21) of indicator_value1
    result['signal_ma11'] = indicator_sma2            # SMA(51) of indicator_value1
    result['indicator_value2'] = indicator_value2     # MA201 - MA301
    
    # Optional: Keep raw value for reference
    result['signal_count_raw'] = indicator_value1
    
    return result
def compute_supertrend(df: pd.DataFrame, period: int, multiplier: float) -> tuple:
    """
    Calculate SuperTrend indicator
    Returns: (supertrend, direction)
    
    FIXED: Better initialization and validation
    """
    if len(df) < period:
        logger.warning(f"Not enough data for SuperTrend period {period}")
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=int)
    
    hl2 = (df['high'] + df['low']) / 2
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    # Initialize first valid values
    supertrend.iloc[:period] = np.nan
    direction.iloc[:period] = 0
    
    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif df['close'].iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
                
            if direction.iloc[i] == 1 and supertrend.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
            if direction.iloc[i] == -1 and supertrend.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
    
    return supertrend, direction


def multi_supertrend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate multiple SuperTrend indicators with different parameters
    Returns DataFrame with directions, flips, and average
    """
    result = pd.DataFrame(index=df.index)
    
    # SuperTrend 7,3
    st_7_3, dir_7_3 = compute_supertrend(df, 7, 3)
    result['st_7_3'] = st_7_3
    result['st_7_3_direction'] = dir_7_3
    result['st_7_3_flip'] = ''
    
    # Detect flips safely
    dir_diff = dir_7_3.diff()
    result.loc[dir_diff == 2, 'st_7_3_flip'] = 'bullish'
    result.loc[dir_diff == -2, 'st_7_3_flip'] = 'bearish'
    
    # SuperTrend 10,2
    st_10_2, dir_10_2 = compute_supertrend(df, 10, 2)
    result['st_10_2'] = st_10_2
    result['st_10_2_direction'] = dir_10_2
    result['st_10_2_flip'] = ''
    
    dir_diff = dir_10_2.diff()
    result.loc[dir_diff == 2, 'st_10_2_flip'] = 'bullish'
    result.loc[dir_diff == -2, 'st_10_2_flip'] = 'bearish'
    
    # SuperTrend 21,7
    st_21_7, dir_21_7 = compute_supertrend(df, 21, 7)
    result['st_21_7'] = st_21_7
    result['st_21_7_direction'] = dir_21_7
    result['st_21_7_flip'] = ''
    
    dir_diff = dir_21_7.diff()
    result.loc[dir_diff == 2, 'st_21_7_flip'] = 'bullish'
    result.loc[dir_diff == -2, 'st_21_7_flip'] = 'bearish'
    
    # Average of all SuperTrend values
    result['st_avg'] = (st_7_3 + st_10_2 + st_21_7) / 3
    
    return result


def resample_to_interval(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Helper function to properly resample dataframe to different interval
    Works with Freqtrade's dataframe structure (uses 'date' column)
    """
    try:
        # Freqtrade uses 'date' column with RangeIndex
        # Set 'date' as index temporarily for resampling
        if 'date' in df.columns:
            df_temp = df.set_index('date')
        else:
            # Fallback: use existing index if it's already datetime
            df_temp = df
        
        df_resampled = df_temp.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in df.columns else 'first'
        }).dropna()
        
        # Reset index to match original structure
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
    except Exception as e:
        logger.error(f"Resampling to {interval} failed: {e}")
        return df.copy()


def compute_rsi_multi_tf(df_1m: pd.DataFrame, rsi_period: int = 14, sma_period: int = 3) -> Dict:
    """
    Compute Weighted Average RSI from multiple timeframes
    
    Logic (matching Pine Script):
    1. Calculate RSI for 1m, 5m, 1h timeframes
    2. Weighted average: avg_rsi = rsi_1m*0.3 + rsi_5m*0.5 + rsi_1h*0.2
    3. Calculate SMA(3) and SMA(11) of avg_rsi
    
    Returns:
    - rsi_1m, rsi_5m, rsi_1h: Individual RSI values
    - avg_rsi: Weighted average RSI
    - avg_rsi_sma: SMA(3) of weighted avg
    - avg_rsi_sma11: SMA(11) of weighted avg
    """
    
    def calc_rsi(df, period):
        """Calculate RSI for a dataframe"""
        df = df.copy()
        df['rsi'] = talib.RSI(df['close'], timeperiod=period)
        return df
    
    # 1-minute RSI (base timeframe)
    df_1m_calc = calc_rsi(df_1m.copy(), rsi_period)
    
    # Resample to 5-minute
    try:
        df_5m = resample_to_interval(df_1m, "5min")
        if len(df_5m) > rsi_period:
            df_5m = calc_rsi(df_5m, rsi_period)
        else:
            logger.warning("Not enough 5m data, using 1m as fallback")
            df_5m = df_1m_calc.copy()
    except Exception as e:
        logger.warning(f"5m resampling failed: {e}, using 1m as fallback")
        df_5m = df_1m_calc.copy()
    
    # Resample to 1-hour
    try:
        df_1h = resample_to_interval(df_1m, "1h")
        if len(df_1h) > rsi_period:
            df_1h = calc_rsi(df_1h, rsi_period)
        else:
            logger.warning("Not enough 1h data, using 5m as fallback")
            df_1h = df_5m.copy()
    except Exception as e:
        logger.warning(f"1h resampling failed: {e}, using 5m as fallback")
        df_1h = df_5m.copy()
    
    # Merge 5m and 1h RSI back to 1m timeframe (forward fill)
    result_df = df_1m_calc.copy()
    
    # Merge 5m RSI
    if 'date' in result_df.columns and 'date' in df_5m.columns:
        result_df = result_df.merge(
            df_5m[['date', 'rsi']].rename(columns={'rsi': 'rsi_5m'}),
            on='date',
            how='left'
        )
        result_df['rsi_5m'] = result_df['rsi_5m'].ffill().fillna(50)
    else:
        result_df['rsi_5m'] = df_5m['rsi'].reindex(result_df.index, method='ffill').fillna(50)
    
    # Merge 1h RSI
    if 'date' in result_df.columns and 'date' in df_1h.columns:
        result_df = result_df.merge(
            df_1h[['date', 'rsi']].rename(columns={'rsi': 'rsi_1h'}),
            on='date',
            how='left'
        )
        result_df['rsi_1h'] = result_df['rsi_1h'].ffill().fillna(50)
    else:
        result_df['rsi_1h'] = df_1h['rsi'].reindex(result_df.index, method='ffill').fillna(50)
    
    # Rename 1m RSI
    result_df['rsi_1m'] = result_df['rsi']
    
    # Calculate weighted average RSI: 0.3*rsi_1m + 0.5*rsi_5m + 0.2*rsi_1h
    result_df['avg_rsi'] = (
        result_df['rsi_1m'] * 0.3 + 
        result_df['rsi_5m'] * 0.5 + 
        result_df['rsi_1h'] * 0.2
    )
    
    # Calculate SMAs of average RSI
    result_df['avg_rsi_sma'] = talib.SMA(result_df['avg_rsi'], timeperiod=sma_period)  # SMA(3)
    result_df['avg_rsi_sma11'] = talib.SMA(result_df['avg_rsi'], timeperiod=11)        # SMA(11)
    
    # Return dictionary with all RSI data
    return {
        "1m": result_df[['rsi_1m', 'avg_rsi', 'avg_rsi_sma', 'avg_rsi_sma11']].rename(
            columns={'rsi_1m': 'rsi', 'avg_rsi_sma': 'rsi_sma'}
        ),
        "5m": result_df[['rsi_5m', 'avg_rsi', 'avg_rsi_sma', 'avg_rsi_sma11']].rename(
            columns={'rsi_5m': 'rsi', 'avg_rsi_sma': 'rsi_sma'}
        ),
        "1h": result_df[['rsi_1h', 'avg_rsi', 'avg_rsi_sma', 'avg_rsi_sma11']].rename(
            columns={'rsi_1h': 'rsi', 'avg_rsi_sma': 'rsi_sma'}
        ),
        "avg": result_df[['avg_rsi', 'avg_rsi_sma', 'avg_rsi_sma11']]
    }

def compute_aroon_multi_tf(df_5m: pd.DataFrame, period: int = 14) -> Dict:
    """
    Computes Aroon Up/Down/Osc for multiple timeframes
    
    FIXED: Simplified resampling approach
    """
    
    def calc_aroon(df):
        df = df.copy()
        df["aroon_up"], df["aroon_down"] = talib.AROON(
            df["high"], df["low"], timeperiod=period
        )
        df["aroon_osc"] = talib.AROONOSC(
            df["high"], df["low"], timeperiod=period
        )
        return df
    
    # 5-minute Aroon
    df_5m_calc = calc_aroon(df_5m.copy())
    
    # 1-hour resampling
    try:
        df_1h = resample_to_interval(df_5m, "1h")
        
        if len(df_1h) > period:
            df_1h = calc_aroon(df_1h)
        else:
            df_1h = df_5m_calc.copy()
            
    except Exception as e:
        logger.warning(f"Aroon 1h resampling failed: {e}")
        df_1h = df_5m_calc.copy()
    
    df_1m = df_5m_calc.copy()
    
    return {
        "1m": df_1m,
        "5m": df_5m_calc,
        "1h": df_1h
    }


def compute_adx_multi_tf(df_5m: pd.DataFrame, adx_period: int = 14) -> Dict:
    """
    Computes ADX, +DI, -DI for multiple timeframes
    
    FIXED: Simplified resampling approach
    """
    
    def calc_adx(df):
        df = df.copy()
        df["plus_di"] = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=adx_period)
        df["minus_di"] = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=adx_period)
        df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=adx_period)
        return df
    
    # 5-minute ADX
    df_5m_calc = calc_adx(df_5m.copy())
    
    # 1-hour resampling
    try:
        df_1h = resample_to_interval(df_5m, "1h")
        
        if len(df_1h) > adx_period:
            df_1h = calc_adx(df_1h)
        else:
            df_1h = df_5m_calc.copy()
            
    except Exception as e:
        logger.warning(f"ADX 1h resampling failed: {e}")
        df_1h = df_5m_calc.copy()
    
    df_1m = df_5m_calc.copy()
    
    return {
        "1m": df_1m,
        "5m": df_5m_calc,
        "1h": df_1h
    }


def compute_macd_multi_tf(df_5m: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """
    Computes MACD, Signal, Histogram for multiple timeframes
    
    FIXED: Simplified resampling approach
    """
    
    def calc_macd(df):
        df = df.copy()
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            df["close"],
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal
        )
        return df
    
    # 5-minute MACD
    df_5m_calc = calc_macd(df_5m.copy())
    
    # 1-hour MACD
    try:
        df_1h = resample_to_interval(df_5m, "1h")
        
        if len(df_1h) > slow:
            df_1h = calc_macd(df_1h)
        else:
            df_1h = df_5m_calc.copy()
            
    except Exception as e:
        logger.warning(f"MACD 1h resampling failed: {e}")
        df_1h = df_5m_calc.copy()
    
    df_1m = df_5m_calc.copy()
    
    return {
        "1m": df_1m,
        "5m": df_5m_calc,
        "1h": df_1h
    }
