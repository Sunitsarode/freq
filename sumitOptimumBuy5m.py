# -*- coding: utf-8 -*-

"""
Sumit Optimum BUY Strategy - Long Only
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


class SumitOptimumBUY5m(IStrategy):
    """
    Long-only strategy with FreqAI integration
    Uses weighted scoring system for entry/exit
    """
    
    # Strategy settings
    timeframe = '5m'
    can_short = False
    startup_candle_count = 320
    process_only_new_candles = True
    
    # ROI settings
    minimal_roi = {
    "0": 0.05
}
    
    # Risk management
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    # Trading mode
    trading_mode = 'futures'
    
    # Exit settings
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    
    # Position management
    max_open_trades = IntParameter(1, 3, default=1, space='buy', optimize=True)
    
    # === SUMIT MA HYPEROPT PARAMETERS ===
    use_sumit_ma = BooleanParameter(default=True, space='buy', optimize=True)
    sumit_ma_weight = IntParameter(1, 3, default=2, space='buy', optimize=True)
    sumit_ma_buy_threshold = DecimalParameter(0, 40, default=30, space='buy', optimize=True)
    sumit_ma_sell_threshold = DecimalParameter(60, 100, default=70, space='buy', optimize=True)
    
    # === RSI HYPEROPT PARAMETERS ===
    use_rsi = BooleanParameter(default=True, space='buy', optimize=True)
    rsi_buy_threshold = IntParameter(30, 50, default=40, space='buy', optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space='buy', optimize=True)
    rsi_weight = IntParameter(1, 3, default=2, space='buy', optimize=True)
    
    # === ADX HYPEROPT PARAMETERS ===
    use_adx = BooleanParameter(default=True, space='buy', optimize=True)
    adx_threshold = IntParameter(20, 35, default=25, space='buy', optimize=True)
    adx_weight = IntParameter(1, 3, default=1, space='buy', optimize=True)
    
    # === SUPERTREND HYPEROPT PARAMETERS ===
    use_supertrend = BooleanParameter(default=True, space='buy', optimize=True)
    st_min_aligned = IntParameter(2, 3, default=2, space='buy', optimize=True)
    st_weight = IntParameter(1, 3, default=2, space='buy', optimize=True)
    
    # === MACD HYPEROPT PARAMETERS ===
    use_macd = BooleanParameter(default=True, space='buy', optimize=True)
    macd_weight = IntParameter(1, 2, default=1, space='buy', optimize=True)
    
    # === SCORING SYSTEM ===
    min_entry_score = IntParameter(6, 12, default=8, space='buy', optimize=True)
    min_exit_score = IntParameter(6, 12, default=7, space='buy', optimize=True)
    
    # FreqAI parameters
    use_freqai = BooleanParameter(default=False, space='buy', optimize=False)
    freqai_threshold = DecimalParameter(0.5, 0.9, default=0.7, space='buy', optimize=False)
    
    # CSV Logging control
    create_backtest_csv = BooleanParameter(default=True, space='buy', optimize=False)
    
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
            "Entry/Exit Score": {
                'entry_score': {'color': 'green', 'type': 'line'},
                'exit_score': {'color': 'red', 'type': 'line'},
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
        """Define LONG entry conditions with weighted scoring"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_tag'] = ''
        dataframe['entry_score'] = 0
        # === SUMIT MA CONDITION ===
               
        if self.use_sumit_ma.value:

            # Condition 1: MA3 < threshold
            sma1_cond = (dataframe['signal_ma3'] < self.sumit_ma_buy_threshold.value)
            dataframe['entry_score'] += sma1_cond.astype(int)

            # Condition 2: MA11 < threshold
            sma2_cond = (dataframe['signal_ma11'] < self.sumit_ma_buy_threshold.value)
            dataframe['entry_score'] += sma2_cond.astype(int)

            # Condition 3: MA3 < MA11 < indicator_value2
            sma_trend_cond = (
                (dataframe['signal_ma3'] < dataframe['signal_ma11']) 
                & (dataframe['signal_ma11'] < dataframe['indicator_value2'])
            )
            dataframe['entry_score']  += sma_trend_cond.astype(int)

        if self.use_rsi.value:

            rsi_low_cond = (dataframe['avg_rsi'] > self.rsi_buy_threshold.value)
            dataframe['entry_score'] += rsi_low_cond.astype(int)

            rsi_rising_cond = (dataframe['avg_rsi_sma'] > dataframe['avg_rsi_sma'].shift(1))
            dataframe['entry_score'] += rsi_rising_cond.astype(int)
        if self.use_adx.value:

            adx_strength = (dataframe['adx_5m'] > self.adx_threshold.value)
            dataframe['entry_score'] += adx_strength.astype(int)

            adx_direction = (dataframe['plus_di_5m'] > dataframe['minus_di_5m'])
            dataframe['entry_score'] += adx_direction.astype(int)
        if self.use_supertrend.value:

            st1 = (dataframe['st_7_3_direction'] == 1).astype(int)
            st2 = (dataframe['st_10_2_direction'] == 1).astype(int)
            st3 = (dataframe['st_21_7_direction'] == 1).astype(int)

            dataframe['entry_score'] += st1 + st2 + st3
        if self.use_macd.value:

            macd_pos = (dataframe['macd_hist_5m'] > 0)
            dataframe['entry_score'] += macd_pos.astype(int)

            macd_rising = (dataframe['macd_hist_5m'] > dataframe['macd_hist_5m'].shift(1))
            dataframe['entry_score'] += macd_rising.astype(int)

        # FreqAI integration (optional)
        if self.use_freqai.value and 'do_predict' in dataframe.columns:
            freqai_cond = dataframe['do_predict'] >= self.freqai_threshold.value
            long_entry = long_entry & freqai_cond

        long_entry = dataframe['entry_score'] >= self.min_entry_score.value
        dataframe.loc[long_entry, 'enter_long'] = 1
        dataframe.loc[long_entry, 'enter_tag'] = 'sumit_long'
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define LONG exit conditions with weighted scoring"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_tag'] = ''
        dataframe['exit_score'] = 0
        

        #   SUMIT MA EXIT (3 points)
        if self.use_sumit_ma.value:

            # Exit Condition 1: MA3 > threshold
            cond1 = (dataframe['signal_ma3'] > self.sumit_ma_sell_threshold.value)
            dataframe['exit_score'] += cond1.astype(int)

            # Exit Condition 2: MA11 > threshold
            cond2 = (dataframe['signal_ma11'] > self.sumit_ma_sell_threshold.value)
            dataframe['exit_score'] += cond2.astype(int)

            # Exit Condition 3: NOT (MA3 < MA11 < indicator_value2)
            cond3 = ~(
                (dataframe['signal_ma3'] < dataframe['signal_ma11']) &
                (dataframe['signal_ma11'] < dataframe['indicator_value2'])
            )
            dataframe['exit_score'] += cond3.astype(int)

    #   RSI EXIT (2 points)
        if self.use_rsi.value:

            # Exit 1: RSI above SELL level (overbought)
            cond_rsi1 = (dataframe['avg_rsi'] > self.rsi_sell_threshold.value)
            dataframe['exit_score'] += cond_rsi1.astype(int)

            # Exit 2: RSI turning down
            cond_rsi2 = (dataframe['avg_rsi_sma'] < dataframe['avg_rsi_sma'].shift(1))
            dataframe['exit_score'] += cond_rsi2.astype(int)
            
    #   ADX EXIT (2 points)
        if self.use_adx.value:

            # Exit 1: Trend weakness
            cond_adx1 = (dataframe['adx_5m'] < self.adx_threshold.value)
            dataframe['exit_score'] += cond_adx1.astype(int)

            # Exit 2: Bearish DI crossover
            cond_adx2 = (dataframe['plus_di_5m'] < dataframe['minus_di_5m'])
            dataframe['exit_score'] += cond_adx2.astype(int)
 #   SUPERTREND EXIT (2 points)
        if self.use_supertrend.value:

            # Each direction gives +1
            dataframe['exit_score'] += (dataframe['st_7_3_direction'] == -1).astype(int)
            dataframe['exit_score'] += (dataframe['st_10_2_direction'] == -1).astype(int)
            dataframe['exit_score'] += (dataframe['st_21_7_direction'] == -1).astype(int)

 #   MACD EXIT (2 points)
        if self.use_macd.value:

            # Exit 1: MACD < Signal (bearish)
            cond_macd1 = (dataframe['macd_5m'] < dataframe['macd_signal_5m'])
            dataframe['exit_score'] += cond_macd1.astype(int)

            # Exit 2: MACD turning down
            cond_macd2 = (
                dataframe['macd_5m'] < dataframe['macd_5m'].shift(1)
            )
            dataframe['exit_score'] += cond_macd2.astype(int)

    # FINAL EXIT CONDITION
        long_exit = dataframe['exit_score'] >= self.min_exit_score.value
        
        dataframe.loc[long_exit, 'exit_long'] = 1
        dataframe.loc[long_exit, 'exit_tag'] = 'sumit_exit'
        
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