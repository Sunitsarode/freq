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


def sumit_ma_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    SUMIT MA SIGNAL LOGIC
    
    Counts BUY and SELL signals based on price position relative to 18 MAs
    (excludes MA 3 High and MA 3 Low)
    
    BUY Signal: Price > MA (counts 0-18)
    SELL Signal: Price < MA (counts 0-18)
    
    Returns DataFrame with buy_signal_count, sell_signal_count, and their MAs
    
    FIXED: Proper .loc indexing to avoid chained assignment warnings
    """
    
    # Calculate OHLC/4 (typical price)
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Define 18 MA periods for OHLC/4 (excluding MA 3 High/Low)
    ma_periods = [3, 9, 15, 21, 27, 31, 37, 51, 65, 81, 101, 121, 131, 151, 171, 201, 251, 301]
    
    # Calculate all 18 MAs
    mas = {}
    for period in ma_periods:
        mas[f'MA_{period}'] = talib.SMA(ohlc4, timeperiod=period)
    
    # Create result DataFrame with proper initialization
    result = pd.DataFrame(index=df.index)
    result['buy_signal_count'] = 0
    result['sell_signal_count'] = 0
    
    # Vectorized calculation instead of row-by-row loop for better performance
    current_price = df['close'].values
    
    for ma_name, ma_series in mas.items():
        ma_values = ma_series.values
        
        # Count where price > MA (buy signal)
        buy_mask = (current_price > ma_values) & pd.notna(ma_values)
        result.loc[buy_mask, 'buy_signal_count'] += 1
        
        # Count where price < MA (sell signal)
        sell_mask = (current_price < ma_values) & pd.notna(ma_values)
        result.loc[sell_mask, 'sell_signal_count'] += 1
    
    # Calculate MA3 and MA11 of signal counts
    result['buy_signal_ma3'] = talib.SMA(result['buy_signal_count'], timeperiod=3)
    result['buy_signal_ma11'] = talib.SMA(result['buy_signal_count'], timeperiod=31)
    result['sell_signal_ma3'] = talib.SMA(result['sell_signal_count'], timeperiod=3)
    result['sell_signal_ma11'] = talib.SMA(result['sell_signal_count'], timeperiod=11)
    
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


def compute_rsi_multi_tf(df_5m: pd.DataFrame, rsi_period: int = 11, sma_period: int = 7) -> Dict:
    """
    Compute RSI for multiple timeframes using proper resampling
    
    FIXED: Simplified resampling that works with Freqtrade's structure
    """
    
    def calc_rsi(df):
        df = df.copy()
        df["avg_price"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        df["rsi"] = talib.RSI(df["avg_price"], timeperiod=rsi_period)
        df["rsi_sma"] = talib.SMA(df["rsi"], timeperiod=sma_period)
        return df
    
    # 5-minute RSI (base timeframe)
    df_5m_calc = calc_rsi(df_5m.copy())
    
    # 1-hour resampling - simplified approach
    try:
        df_1h = resample_to_interval(df_5m, "1h")
        
        if len(df_1h) > rsi_period:
            df_1h = calc_rsi(df_1h)
        else:
            logger.warning("Not enough 1h data after resampling, using 5m as fallback")
            df_1h = df_5m_calc.copy()
            
    except Exception as e:
        logger.warning(f"1h resampling failed: {e}, using 5m data as fallback")
        df_1h = df_5m_calc.copy()
    
    # For 1m, use 5m data (limitation of working with 5m timeframe)
    df_1m = df_5m_calc.copy()
    
    return {
        "1m": df_1m,
        "5m": df_5m_calc,
        "1h": df_1h
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
