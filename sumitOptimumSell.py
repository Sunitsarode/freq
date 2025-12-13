# -*- coding: utf-8 -*-

"""
Sumit Optimum SELL Strategy - Short Only
Optimized multi-indicator strategy with FreqAI integration
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from pandas import DataFrame
import pandas as pd
import talib
from datetime import datetime
import logging
from typing import Optional
import sys
from pathlib import Path

# Add sumit_ma folder to Python path
strategy_path = Path(__file__).parent / 'sumit_ma'
sys.path.insert(0, str(strategy_path))

# Import custom indicators
from indicators import (
    sumit_ma_signals,
    multi_supertrend,
    compute_rsi_multi_tf,
    compute_adx_multi_tf,
    compute_macd_multi_tf
)

# Import backtesting logger
from backtest_logger import BacktestLogger

logger = logging.getLogger(__name__)


class SumitOptimumSELL(IStrategy):
    """
    Short-only strategy with FreqAI integration
    Uses weighted scoring system for entry/exit
    """
    
    # Strategy settings
    timeframe = '1m'
    can_short = True
    startup_candle_count = 320
    process_only_new_candles = True
    
    # ROI settings
    minimal_roi = {
        "0": 0.03,
        "60": 0.025,
        "180": 0.02,
        "360": 0.015
    }
    
    # Risk management
    stoploss = -0.035
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    
    # Trading mode
    trading_mode = 'futures'
    
    # Exit settings
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    
    # Position management
    max_open_trades = IntParameter(1, 3, default=1, space='sell', optimize=True)
    
    # === SUMIT MA HYPEROPT PARAMETERS ===
    use_sumit_ma = BooleanParameter(default=True, space='sell', optimize=True)
    sumit_ma_weight = IntParameter(1, 3, default=2, space='sell', optimize=True)
    
    # === RSI HYPEROPT PARAMETERS ===
    use_rsi = BooleanParameter(default=True, space='sell', optimize=True)
    rsi_buy_threshold = IntParameter(30, 50, default=40, space='sell', optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space='sell', optimize=True)
    rsi_weight = IntParameter(1, 3, default=2, space='sell', optimize=True)
    
    # === ADX HYPEROPT PARAMETERS ===
    use_adx = BooleanParameter(default=True, space='sell', optimize=True)
    adx_threshold = IntParameter(20, 35, default=25, space='sell', optimize=True)
    adx_weight = IntParameter(1, 3, default=1, space='sell', optimize=True)
    
    # === SUPERTREND HYPEROPT PARAMETERS ===
    use_supertrend = BooleanParameter(default=True, space='sell', optimize=True)
    st_min_aligned = IntParameter(2, 3, default=2, space='sell', optimize=True)
    st_weight = IntParameter(1, 3, default=2, space='sell', optimize=True)
    
    # === MACD HYPEROPT PARAMETERS ===
    use_macd = BooleanParameter(default=True, space='sell', optimize=True)
    macd_weight = IntParameter(1, 2, default=1, space='sell', optimize=True)
    
    # === SCORING SYSTEM ===
    min_entry_score = IntParameter(4, 8, default=5, space='sell', optimize=True)
    min_exit_score = IntParameter(2, 4, default=2, space='sell', optimize=True)
    
    # FreqAI parameters
    use_freqai = BooleanParameter(default=False, space='sell', optimize=False)
    freqai_threshold = DecimalParameter(0.5, 0.9, default=0.7, space='sell', optimize=False)
    
    # Store for entry indicators
    entry_indicators = {}
    
    # Backtesting logger
    backtest_logger = None
    
    def __init__(self, config: dict) -> None:
        """Initialize strategy with backtesting logger"""
        super().__init__(config)
        
        if config.get('runmode') != 'hyperopt':
            try:
                self.backtest_logger = BacktestLogger(strategy_name=self.__class__.__name__)
                logger.info(">> Backtest logger initialized")
            except Exception as e:
                logger.warning(f">> Backtest logger initialization failed: {e}")
                self.backtest_logger = None
        else:
            self.backtest_logger = None
            logger.info(">> Backtest logger disabled for hyperopt")
    
    # Plot configuration
    plot_config = {
        'main_plot': {
            'st_7_3': {'color': 'blue', 'type': 'line'},
            'st_10_2': {'color': 'orange', 'type': 'line'},
            'st_21_7': {'color': 'red', 'type': 'line'},
        },
        'subplots': {
            "Sumit MA": {
                'signal_count': {'color': 'blue', 'type': 'line'},
                'signal_ma3': {'color': 'purple', 'type': 'line'},
                'signal_ma11': {'color': 'green', 'type': 'line'},
            },
            "RSI": {
                'avg_rsi': {'color': 'blue', 'type': 'line'},
                'avg_rsi_sma': {'color': 'purple', 'type': 'line'},
            },
            "ADX": {
                'adx_5m': {'color': 'purple', 'type': 'line'},
            },
        }
    }
    
    def informative_pairs(self):
        """Define additional pairs/timeframes"""
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all indicators"""
        if len(dataframe) < self.startup_candle_count:
            logger.warning(f"Not enough data: {len(dataframe)} < {self.startup_candle_count}")
            return dataframe
        
        df = dataframe.copy()
        
        try:
            # 1. SUMIT MA SIGNALS
            ma_signals = sumit_ma_signals(df)
            for col in ma_signals.columns:
                df[col] = ma_signals[col]
            
            # 2. MULTI SUPERTREND
            st_signals = multi_supertrend(df)
            for col in st_signals.columns:
                df[col] = st_signals[col]
            
            # 3. RSI MULTI-TIMEFRAME with weighted average
            rsi_data = compute_rsi_multi_tf(df)
            
            # Get avg_rsi and its SMAs
            if 'avg' in rsi_data:
                df['avg_rsi'] = rsi_data['avg']['avg_rsi']
                df['avg_rsi_sma'] = rsi_data['avg']['avg_rsi_sma']
                df['avg_rsi_sma11'] = rsi_data['avg']['avg_rsi_sma11']
            
            # Also keep individual RSI values for reference
            df['rsi_1m'] = rsi_data['1m']['rsi']
            df['rsi_5m'] = rsi_data['5m']['rsi']
            df['rsi_1h'] = rsi_data['1h']['rsi']
            
            # Fill NaN values
            df['avg_rsi'] = df['avg_rsi'].fillna(50)
            df['avg_rsi_sma'] = df['avg_rsi_sma'].fillna(50)
            df['avg_rsi_sma11'] = df['avg_rsi_sma11'].fillna(50)
            
            # 4. ADX MULTI-TIMEFRAME
            adx_data = compute_adx_multi_tf(df)
            df['adx_5m'] = adx_data['5m']['adx']
            df['plus_di_5m'] = adx_data['5m']['plus_di']
            df['minus_di_5m'] = adx_data['5m']['minus_di']
            
            # 5. MACD MULTI-TIMEFRAME
            macd_data = compute_macd_multi_tf(df)
            df['macd_5m'] = macd_data['5m']['macd']
            df['macd_signal_5m'] = macd_data['5m']['macd_signal']
            df['macd_hist_5m'] = macd_data['5m']['macd_hist']
            
            # Fill NaN values in critical columns
            fill_columns = [
                'signal_count', 'signal_ma3', 'signal_ma11', 'indicator_value2',
                'st_7_3_direction', 'st_10_2_direction', 'st_21_7_direction',
                'avg_rsi', 'adx_5m', 'macd_hist_5m'
            ]
            
            for col in fill_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            import traceback
            traceback.print_exc()
        
        return df
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define SHORT entry conditions with weighted scoring"""
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''
        
        conditions = []
        score = 0
        
        # === SUMIT MA CONDITION ===
        if self.use_sumit_ma.value:
            # SELL Enter: indicator_sma1 > 70% & indicator_sma2 > 70% & sma1 < sma2 < indicator_value2
            sumit_ma_cond = (
                (dataframe['signal_ma3'] > dataframe['price_70']) &
                (dataframe['signal_ma11'] > dataframe['price_70']) &
                (dataframe['signal_ma3'] < dataframe['signal_ma11']) &
                (dataframe['signal_ma11'] < dataframe['indicator_value2'])
            )
            conditions.append(sumit_ma_cond)
            score = score + (sumit_ma_cond.astype(int) * self.sumit_ma_weight.value)
        
        # === RSI CONDITION ===
        if self.use_rsi.value:
            # Use weighted average RSI - overbought for short
            rsi_cond = (
                (dataframe['avg_rsi'] > self.rsi_sell_threshold.value) &
                (dataframe['avg_rsi_sma'] < dataframe['avg_rsi_sma'].shift(1))  # Falling
            )
            conditions.append(rsi_cond)
            score = score + (rsi_cond.astype(int) * self.rsi_weight.value)
        
        # === ADX CONDITION ===
        if self.use_adx.value:
            adx_cond = (
                (dataframe['adx_5m'] > self.adx_threshold.value) &
                (dataframe['minus_di_5m'] > dataframe['plus_di_5m'])
            )
            conditions.append(adx_cond)
            score = score + (adx_cond.astype(int) * self.adx_weight.value)
        
        # === SUPERTREND CONDITION ===
        if self.use_supertrend.value:
            st_aligned_count = (
                (dataframe['st_7_3_direction'] == -1).astype(int) +
                (dataframe['st_10_2_direction'] == -1).astype(int) +
                (dataframe['st_21_7_direction'] == -1).astype(int)
            )
            st_cond = st_aligned_count >= self.st_min_aligned.value
            conditions.append(st_cond)
            score = score + (st_cond.astype(int) * self.st_weight.value)
        
        # === MACD CONDITION ===
        if self.use_macd.value:
            macd_cond = (
                (dataframe['macd_hist_5m'] < 0) &
                (dataframe['macd_hist_5m'] < dataframe['macd_hist_5m'].shift(1))
            )
            conditions.append(macd_cond)
            score = score + (macd_cond.astype(int) * self.macd_weight.value)
        
        # Entry based on score
        short_entry = score >= self.min_entry_score.value
        
        # FreqAI integration (optional)
        if self.use_freqai.value and 'do_predict' in dataframe.columns:
            freqai_cond = dataframe['do_predict'] <= (1 - self.freqai_threshold.value)
            short_entry = short_entry & freqai_cond
        
        dataframe.loc[short_entry, 'enter_short'] = 1
        dataframe.loc[short_entry, 'enter_tag'] = 'sumit_short'
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define SHORT exit conditions with weighted scoring"""
        dataframe.loc[:, 'exit_short'] = 0
        dataframe.loc[:, 'exit_tag'] = ''
        
        exit_score = 0
        
        # === SUMIT MA EXIT ===
        if self.use_sumit_ma.value:
            # SELL Exit: indicator_sma1 < 40% | indicator_sma2 < 40% | NOT(sma1 < sma2 < indicator_value2)
            sumit_ma_exit = (
                (dataframe['signal_ma3'] < dataframe['price_40']) |
                (dataframe['signal_ma11'] < dataframe['price_40']) |
                ~((dataframe['signal_ma3'] < dataframe['signal_ma11']) & 
                  (dataframe['signal_ma11'] < dataframe['indicator_value2']))
            )
            exit_score = exit_score + (sumit_ma_exit.astype(int) * self.sumit_ma_weight.value)
        
        # === RSI EXIT ===
        if self.use_rsi.value:
            rsi_exit = (
                (dataframe['avg_rsi'] < self.rsi_buy_threshold.value) |
                (dataframe['avg_rsi_sma'] > dataframe['avg_rsi_sma'].shift(1))  # Rising
            )
            exit_score = exit_score + (rsi_exit.astype(int) * self.rsi_weight.value)
        
        # === ADX EXIT ===
        if self.use_adx.value:
            adx_exit = dataframe['minus_di_5m'] < dataframe['plus_di_5m']
            exit_score = exit_score + (adx_exit.astype(int) * self.adx_weight.value)
        
        # === SUPERTREND EXIT ===
        if self.use_supertrend.value:
            st_exit = (
                (dataframe['st_7_3_direction'] == 1) &
                (dataframe['st_10_2_direction'] == 1)
            )
            exit_score = exit_score + (st_exit.astype(int) * self.st_weight.value)
        
        # === MACD EXIT ===
        if self.use_macd.value:
            macd_exit = (
                (dataframe['macd_5m'] > dataframe['macd_signal_5m']) &
                (dataframe['macd_5m'].shift(1) <= dataframe['macd_signal_5m'].shift(1))
            )
            exit_score = exit_score + (macd_exit.astype(int) * self.macd_weight.value)
        
        # Exit based on score
        short_exit = exit_score >= self.min_exit_score.value
        
        dataframe.loc[short_exit, 'exit_short'] = 1
        dataframe.loc[short_exit, 'exit_tag'] = 'sumit_exit'
        
        return dataframe
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """Store entry indicators and log to backtest CSV"""
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) > 0:
                last_candle = dataframe.iloc[-1]
                
                indicators = {
                    'signal_count': last_candle.get('signal_count', 0),
                    'avg_rsi': last_candle.get('avg_rsi', 0),
                    'adx_5m': last_candle.get('adx_5m', 0),
                    'st_7_3_direction': last_candle.get('st_7_3_direction', 0),
                    'st_10_2_direction': last_candle.get('st_10_2_direction', 0),
                    'st_21_7_direction': last_candle.get('st_21_7_direction', 0),
                    'macd_hist_5m': last_candle.get('macd_hist_5m', 0),
                }
                
                self.entry_indicators[pair] = {
                    'entry_time': current_time,
                    'entry_price': rate,
                    'side': side,
                    **indicators
                }
                
                if self.backtest_logger and hasattr(self.backtest_logger, 'log_entry'):
                    self.backtest_logger.log_entry(
                        pair=pair,
                        entry_time=current_time,
                        entry_price=rate,
                        side=side,
                        indicators=indicators
                    )
        
        except Exception as e:
            logger.error(f"Error in confirm_trade_entry: {e}")
        
        return True
    
    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """Log exit to backtest CSV"""
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) > 0:
                last_candle = dataframe.iloc[-1]
                
                profit_ratio = trade.calc_profit_ratio(rate)
                
                exit_indicators = {
                    'signal_count': last_candle.get('signal_count', 0),
                    'avg_rsi': last_candle.get('avg_rsi', 0),
                    'adx_5m': last_candle.get('adx_5m', 0),
                    'st_7_3_direction': last_candle.get('st_7_3_direction', 0),
                    'st_10_2_direction': last_candle.get('st_10_2_direction', 0),
                    'st_21_7_direction': last_candle.get('st_21_7_direction', 0),
                    'macd_hist_5m': last_candle.get('macd_hist_5m', 0),
                }
                
                if self.backtest_logger:
                    self.backtest_logger.log_exit(
                        pair=pair,
                        exit_time=current_time,
                        exit_price=rate,
                        profit_ratio=profit_ratio,
                        exit_reason=exit_reason,
                        indicators=exit_indicators
                    )
                
                if pair in self.entry_indicators:
                    del self.entry_indicators[pair]
        
        except Exception as e:
            logger.error(f"Error in confirm_trade_exit: {e}")
        
        return True
