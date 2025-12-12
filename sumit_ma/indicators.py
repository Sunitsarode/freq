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
    SUMIT MA SIGNAL LOGIC - WITH TRANSITION ZONE
    
    Score range: +20 to -20
    
    Logic:
    1. If price > MA_301: score = +number of MAs above it
    2. If price < MA_251: score = -number of MAs below it
    3. If MA_251 < price < MA_301: score = +1 (transition zone - weak buy)
    4. If MA_301 < price < MA_251: score = -1 (transition zone - weak sell)
    
    MA Order:
    [MA_3_low, MA_3_high, MA_3, MA_9, MA_15, MA_21, MA_27, MA_31, MA_37, MA_51,
     MA_65, MA_81, MA_101, MA_121, MA_131, MA_151, MA_171, MA_201, MA_251, MA_301]
    """
    
    # Typical price
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # EXACT MA order (20 MAs)
    ma_list = [
        ('MA_3_low', talib.SMA(df['low'], timeperiod=3)),
        ('MA_3_high', talib.SMA(df['high'], timeperiod=3)),
    ]
    
    # Remaining 18 MAs on OHLC/4
    ma_periods = [3, 9, 15, 21, 27, 31, 37, 51, 65, 81,
                  101, 121, 131, 151, 171, 201, 251, 301]
    for p in ma_periods:
        ma_list.append((f'MA_{p}', talib.SMA(ohlc4, timeperiod=p)))
    
    # Combine all MA series into 2D numpy array shape: (rows, 20)
    ma_matrix = np.column_stack([ma.values for _, ma in ma_list])
    
    # Price array
    price = df['close'].values
    
    # Extract MA_251 (index 18) and MA_301 (index 19)
    ma_251 = ma_matrix[:, 18]
    ma_301 = ma_matrix[:, 19]
    
    # Count MAs that price is above
    mas_above = (price.reshape(-1, 1) > ma_matrix).sum(axis=1)  # 0 to 20
    
    # Count MAs that price is below
    mas_below = (price.reshape(-1, 1) < ma_matrix).sum(axis=1)  # 0 to 20
    
    # Initialize signal array
    signal_count = np.zeros(len(price))
    
    # Apply conditions
    for i in range(len(price)):
        if price[i] > ma_301[i]:
            # Price above MA_301 → positive score
            signal_count[i] = mas_above[i]
        elif price[i] < ma_251[i]:
            # Price below MA_251 → negative score
            signal_count[i] = -mas_below[i]
        elif ma_251[i] < price[i] < ma_301[i]:
            # Between MA_251 and MA_301 (MA_251 lower) → weak buy
            signal_count[i] = 1
        elif ma_301[i] < price[i] < ma_251[i]:
            # Between MA_301 and MA_251 (MA_301 lower) → weak sell
            signal_count[i] = -1
        else:
            # Price exactly at MA (edge case)
            signal_count[i] = 0
    
    # Build output DataFrame
    result = pd.DataFrame(index=df.index)
    result['signal_count'] = signal_count
    
    # Moving averages of the signal
    result['signal_ma3'] = talib.SMA(result['signal_count'], timeperiod=7)
    result['signal_ma11'] = talib.SMA(result['signal_count'], timeperiod=21)
    
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
