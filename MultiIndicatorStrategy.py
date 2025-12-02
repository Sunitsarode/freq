"""
Multi-Indicator Strategy for Freqtrade
UPDATED: Added backtesting logger integration
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from pandas import DataFrame
import pandas as pd
import talib
from datetime import datetime
import logging
from typing import Optional
import json
import sys
from pathlib import Path

# Add sumit_ma folder to Python path
strategy_path = Path(__file__).parent / 'sumit_ma'
sys.path.insert(0, str(strategy_path))

# Import custom indicators from sumit_ma folder
from indicators import (
    sumit_ma_signals,
    multi_supertrend,
    compute_rsi_multi_tf,
    compute_aroon_multi_tf,
    compute_adx_multi_tf,
    compute_macd_multi_tf
)

# Import notification and CSV utilities from sumit_ma folder
from utils import send_ntfy_notification, save_trade_to_csv

# Import backtesting logger
from backtest_logger import BacktestLogger

logger = logging.getLogger(__name__)


class MultiIndicatorStrategy(IStrategy):
    """
    Advanced multi-indicator strategy with comprehensive signal generation
    UPDATED: Integrated backtesting logger
    """
    
    # Strategy settings
    timeframe = '5m'
    can_short = True
    startup_candle_count = 320
    process_only_new_candles = True
    
    # Risk management
    stoploss = -0.02
    
    # Trading mode
    trading_mode = 'futures'
    
    # Exit settings
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    
    # Position management
    max_open_trades = IntParameter(1, 5, default=3, space='buy', optimize=True)
    
    # Optional features
    enable_notifications = BooleanParameter(default=False, space='buy', optimize=False)
    enable_csv_logging = BooleanParameter(default=False, space='buy', optimize=False)
    ntfy_topic = 'freqtrade_alerts'
    
    # Hyperopt parameters - Sumit MA Signals
    buy_ma_cross_threshold = IntParameter(5, 15, default=10, space='buy', optimize=True)
    sell_ma_cross_threshold = IntParameter(5, 15, default=10, space='buy', optimize=True)
    
    # Hyperopt parameters - SuperTrend
    use_supertrend = BooleanParameter(default=True, space='buy', optimize=True)
    st_min_aligned = IntParameter(1, 3, default=2, space='buy', optimize=True)
    
    # Hyperopt parameters - RSI
    rsi_5m_buy_threshold = IntParameter(25, 45, default=35, space='buy', optimize=True)
    rsi_5m_sell_threshold = IntParameter(55, 75, default=65, space='buy', optimize=True)
    rsi_1h_buy_threshold = IntParameter(30, 50, default=40, space='buy', optimize=True)
    rsi_1h_sell_threshold = IntParameter(50, 70, default=60, space='buy', optimize=True)
    
    # Hyperopt parameters - ADX
    adx_threshold = IntParameter(15, 30, default=20, space='buy', optimize=True)
    
    # Hyperopt parameters - Aroon
    aroon_osc_buy = IntParameter(30, 70, default=50, space='buy', optimize=True)
    aroon_osc_sell = IntParameter(-70, -30, default=-50, space='buy', optimize=True)
    
    # Entry logic type
    require_all_conditions = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Store for entry indicators
    entry_indicators = {}
    
    # Backtesting logger (initialized in __init__)
    backtest_logger = None
    
    def __init__(self, config: dict) -> None:
        """Initialize strategy with backtesting logger"""
        super().__init__(config)
        
        # Initialize backtest logger
        try:
            self.backtest_logger = BacktestLogger(strategy_name=self.__class__.__name__)
            logger.info("✅ Backtest logger initialized")
        except Exception as e:
            logger.warning(f"⚠️ Backtest logger initialization failed: {e}")
            self.backtest_logger = None
    
    # Plot configuration
    plot_config = {
        'main_plot': {
            'st_7_3': {'color': 'blue', 'type': 'line'},
            'st_10_2': {'color': 'orange', 'type': 'line'},
            'st_21_7': {'color': 'red', 'type': 'line'},
            'st_avg': {'color': 'purple', 'type': 'line'},
        },
        'subplots': {
            "MA Signals": {
                'buy_signal_count': {'color': 'green', 'type': 'line'},
                'sell_signal_count': {'color': 'red', 'type': 'line'},
                'buy_signal_ma3': {'color': 'lightgreen', 'type': 'line'},
                'buy_signal_ma11': {'color': 'darkgreen', 'type': 'line'},
            },
            "RSI": {
                'rsi_5m': {'color': 'blue', 'type': 'line'},
                'rsi_1h': {'color': 'orange', 'type': 'line'},
            },
            "ADX": {
                'adx_5m': {'color': 'purple', 'type': 'line'},
                'plus_di_5m': {'color': 'green', 'type': 'line'},
                'minus_di_5m': {'color': 'red', 'type': 'line'},
            },
            "MACD": {
                'macd_5m': {'color': 'blue', 'type': 'line'},
                'macd_signal_5m': {'color': 'orange', 'type': 'line'},
                'macd_hist_5m': {'color': 'gray', 'type': 'bar'},
            },
            "Aroon": {
                'aroon_up_5m': {'color': 'green', 'type': 'line'},
                'aroon_down_5m': {'color': 'red', 'type': 'line'},
                'aroon_osc_5m': {'color': 'blue', 'type': 'line'},
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
            
            # 3. RSI MULTI-TIMEFRAME
            rsi_data = compute_rsi_multi_tf(df)
            df['rsi_5m'] = rsi_data['5m']['rsi']
            df['rsi_5m_sma'] = rsi_data['5m']['rsi_sma']
            
            if len(rsi_data['1h']) > 0 and 'date' in df.columns and 'date' in rsi_data['1h'].columns:
                df = df.merge(
                    rsi_data['1h'][['date', 'rsi', 'rsi_sma']].rename(
                        columns={'rsi': 'rsi_1h', 'rsi_sma': 'rsi_1h_sma'}
                    ),
                    on='date',
                    how='left'
                )
                df['rsi_1h'] = df['rsi_1h'].ffill()
                df['rsi_1h_sma'] = df['rsi_1h_sma'].ffill()
            else:
                df['rsi_1h'] = rsi_data['5m']['rsi']
                df['rsi_1h_sma'] = rsi_data['5m']['rsi_sma']
            
            df['rsi_1h'] = df['rsi_1h'].fillna(50)
            df['rsi_1h_sma'] = df['rsi_1h_sma'].fillna(50)
            
            # 4. AROON MULTI-TIMEFRAME
            aroon_data = compute_aroon_multi_tf(df)
            df['aroon_up_5m'] = aroon_data['5m']['aroon_up']
            df['aroon_down_5m'] = aroon_data['5m']['aroon_down']
            df['aroon_osc_5m'] = aroon_data['5m']['aroon_osc']
            
            # 5. ADX MULTI-TIMEFRAME
            adx_data = compute_adx_multi_tf(df)
            df['adx_5m'] = adx_data['5m']['adx']
            df['plus_di_5m'] = adx_data['5m']['plus_di']
            df['minus_di_5m'] = adx_data['5m']['minus_di']
            
            # 6. MACD MULTI-TIMEFRAME
            macd_data = compute_macd_multi_tf(df)
            df['macd_5m'] = macd_data['5m']['macd']
            df['macd_signal_5m'] = macd_data['5m']['macd_signal']
            df['macd_hist_5m'] = macd_data['5m']['macd_hist']
            
            # Fill NaN values in critical columns
            fill_columns = [
                'buy_signal_count', 'sell_signal_count', 
                'buy_signal_ma3', 'buy_signal_ma11',
                'sell_signal_ma3', 'sell_signal_ma11',
                'st_7_3_direction', 'st_10_2_direction', 'st_21_7_direction',
                'rsi_5m', 'rsi_1h', 'adx_5m', 
                'aroon_osc_5m', 'macd_hist_5m'
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
        """Define entry conditions"""
        conditions_long = []
        conditions_short = []
        
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''
        
        # === LONG ENTRY CONDITIONS ===
        
        ma_cross_long = (
            (dataframe['buy_signal_ma3'] > dataframe['buy_signal_ma11']) &
            (dataframe['buy_signal_ma3'].shift(1) <= dataframe['buy_signal_ma11'].shift(1)) &
            (dataframe['buy_signal_count'] >= self.buy_ma_cross_threshold.value)
        )
        conditions_long.append(ma_cross_long)
        
        if self.use_supertrend.value:
            st_aligned_count = (
                (dataframe['st_7_3_direction'] == 1).astype(int) +
                (dataframe['st_10_2_direction'] == 1).astype(int) +
                (dataframe['st_21_7_direction'] == 1).astype(int)
            )
            st_bullish = st_aligned_count >= self.st_min_aligned.value
            conditions_long.append(st_bullish)
        
        rsi_long = (
            (dataframe['rsi_5m'] > self.rsi_5m_buy_threshold.value) &
            (dataframe['rsi_1h'] > self.rsi_1h_buy_threshold.value)
        )
        conditions_long.append(rsi_long)
        
        adx_strong = dataframe['adx_5m'] > self.adx_threshold.value
        conditions_long.append(adx_strong)
        
        aroon_long = dataframe['aroon_osc_5m'] > self.aroon_osc_buy.value
        conditions_long.append(aroon_long)
        
        macd_long = dataframe['macd_hist_5m'] > 0
        conditions_long.append(macd_long)
        
        if self.require_all_conditions.value:
            long_entry = conditions_long[0]
            for condition in conditions_long[1:]:
                long_entry = long_entry & condition
        else:
            score = sum(c.astype(int) for c in conditions_long)
            long_entry = score >= 4
        
        dataframe.loc[long_entry, 'enter_long'] = 1
        dataframe.loc[long_entry, 'enter_tag'] = 'multi_signal_long'
        
        # === SHORT ENTRY CONDITIONS ===
        if self.can_short:
            ma_cross_short = (
                (dataframe['sell_signal_ma3'] > dataframe['sell_signal_ma11']) &
                (dataframe['sell_signal_ma3'].shift(1) <= dataframe['sell_signal_ma11'].shift(1)) &
                (dataframe['sell_signal_count'] >= self.sell_ma_cross_threshold.value)
            )
            conditions_short.append(ma_cross_short)
            
            if self.use_supertrend.value:
                st_aligned_count = (
                    (dataframe['st_7_3_direction'] == -1).astype(int) +
                    (dataframe['st_10_2_direction'] == -1).astype(int) +
                    (dataframe['st_21_7_direction'] == -1).astype(int)
                )
                st_bearish = st_aligned_count >= self.st_min_aligned.value
                conditions_short.append(st_bearish)
            
            rsi_short = (
                (dataframe['rsi_5m'] < self.rsi_5m_sell_threshold.value) &
                (dataframe['rsi_1h'] < self.rsi_1h_sell_threshold.value)
            )
            conditions_short.append(rsi_short)
            
            adx_strong_short = dataframe['adx_5m'] > self.adx_threshold.value
            conditions_short.append(adx_strong_short)
            
            aroon_short = dataframe['aroon_osc_5m'] < self.aroon_osc_sell.value
            conditions_short.append(aroon_short)
            
            macd_short = dataframe['macd_hist_5m'] < 0
            conditions_short.append(macd_short)
            
            if self.require_all_conditions.value:
                short_entry = conditions_short[0]
                for condition in conditions_short[1:]:
                    short_entry = short_entry & condition
            else:
                score = sum(c.astype(int) for c in conditions_short)
                short_entry = score >= 4
            
            dataframe.loc[short_entry, 'enter_short'] = 1
            dataframe.loc[short_entry, 'enter_tag'] = 'multi_signal_short'
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        dataframe.loc[:, 'exit_tag'] = ''
        
        # === EXIT LONG ===
        
        ma_reversal_long = (
            (dataframe['sell_signal_ma3'] > dataframe['buy_signal_ma3']) &
            (dataframe['sell_signal_count'] >= self.sell_ma_cross_threshold.value)
        )
        
        rsi_overbought = dataframe['rsi_5m'] > 80
        
        macd_cross_down = (
            (dataframe['macd_5m'] < dataframe['macd_signal_5m']) &
            (dataframe['macd_5m'].shift(1) >= dataframe['macd_signal_5m'].shift(1))
        )
        
        st_flip_bearish = (
            (dataframe['st_7_3_direction'] == -1) &
            (dataframe['st_10_2_direction'] == -1)
        )
        
        exit_score_long = (
            ma_reversal_long.astype(int) +
            rsi_overbought.astype(int) +
            macd_cross_down.astype(int) +
            st_flip_bearish.astype(int)
        )
        
        exit_long = exit_score_long >= 2
        dataframe.loc[exit_long, 'exit_long'] = 1
        dataframe.loc[exit_long, 'exit_tag'] = 'multi_exit_signal'
        
        # === EXIT SHORT ===
        if self.can_short:
            ma_reversal_short = (
                (dataframe['buy_signal_ma3'] > dataframe['sell_signal_ma3']) &
                (dataframe['buy_signal_count'] >= self.buy_ma_cross_threshold.value)
            )
            
            rsi_oversold = dataframe['rsi_5m'] < 20
            
            macd_cross_up = (
                (dataframe['macd_5m'] > dataframe['macd_signal_5m']) &
                (dataframe['macd_5m'].shift(1) <= dataframe['macd_signal_5m'].shift(1))
            )
            
            st_flip_bullish = (
                (dataframe['st_7_3_direction'] == 1) &
                (dataframe['st_10_2_direction'] == 1)
            )
            
            exit_score_short = (
                ma_reversal_short.astype(int) +
                rsi_oversold.astype(int) +
                macd_cross_up.astype(int) +
                st_flip_bullish.astype(int)
            )
            
            exit_short = exit_score_short >= 2
            dataframe.loc[exit_short, 'exit_short'] = 1
            dataframe.loc[exit_short, 'exit_tag'] = 'multi_exit_signal'
        
        return dataframe
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """Store entry indicators and log to backtest CSV"""
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) > 0:
                last_candle = dataframe.iloc[-1]
                
                # Prepare indicator dictionary
                indicators = {
                    'buy_signal_count': last_candle.get('buy_signal_count', 0),
                    'sell_signal_count': last_candle.get('sell_signal_count', 0),
                    'rsi_5m': last_candle.get('rsi_5m', 0),
                    'rsi_1h': last_candle.get('rsi_1h', 0),
                    'adx_5m': last_candle.get('adx_5m', 0),
                    'st_7_3_direction': last_candle.get('st_7_3_direction', 0),
                    'st_10_2_direction': last_candle.get('st_10_2_direction', 0),
                    'st_21_7_direction': last_candle.get('st_21_7_direction', 0),
                    'aroon_osc_5m': last_candle.get('aroon_osc_5m', 0),
                    'macd_hist_5m': last_candle.get('macd_hist_5m', 0),
                }
                
                # Store for old CSV logging system
                self.entry_indicators[pair] = {
                    'entry_time': current_time,
                    'entry_price': rate,
                    'side': side,
                    **indicators
                }
                
                # Log to backtest CSV
                if self.backtest_logger:
                    self.backtest_logger.log_entry(
                        pair=pair,
                        entry_time=current_time,
                        entry_price=rate,
                        side=side,
                        indicators=indicators
                    )
                
                # Send notification if enabled
                if self.enable_notifications.value:
                    message = f"🟢 ENTRY {side.upper()}\n"
                    message += f"Pair: {pair}\n"
                    message += f"Price: {rate:.8f}\n"
                    message += f"Time: {current_time}\n"
                    message += f"Buy Signals: {indicators['buy_signal_count']}\n"
                    message += f"Sell Signals: {indicators['sell_signal_count']}\n"
                    message += f"RSI 5m: {indicators['rsi_5m']:.2f}\n"
                    message += f"ADX 5m: {indicators['adx_5m']:.2f}"
                    
                    send_ntfy_notification(message, self.ntfy_topic, priority=4)
        
        except Exception as e:
            logger.error(f"Error in confirm_trade_entry: {e}")
        
        return True
    
    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """Log exit and send notifications"""
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) > 0:
                last_candle = dataframe.iloc[-1]
                entry_data = self.entry_indicators.get(pair, {})
                
                profit_ratio = trade.calc_profit_ratio(rate)
                profit_abs = trade.calc_profit(rate)
                
                # Prepare exit indicators
                exit_indicators = {
                    'buy_signal_count': last_candle.get('buy_signal_count', 0),
                    'sell_signal_count': last_candle.get('sell_signal_count', 0),
                    'rsi_5m': last_candle.get('rsi_5m', 0),
                    'rsi_1h': last_candle.get('rsi_1h', 0),
                    'adx_5m': last_candle.get('adx_5m', 0),
                    'st_7_3_direction': last_candle.get('st_7_3_direction', 0),
                    'st_10_2_direction': last_candle.get('st_10_2_direction', 0),
                    'st_21_7_direction': last_candle.get('st_21_7_direction', 0),
                    'aroon_osc_5m': last_candle.get('aroon_osc_5m', 0),
                    'macd_hist_5m': last_candle.get('macd_hist_5m', 0),
                }
                
                # Log to backtest CSV
                if self.backtest_logger:
                    self.backtest_logger.log_exit(
                        pair=pair,
                        exit_time=current_time,
                        exit_price=rate,
                        profit_ratio=profit_ratio,
                        exit_reason=exit_reason,
                        indicators=exit_indicators
                    )
                
                # Old CSV logging if enabled
                if self.enable_csv_logging.value:
                    trade_data = {
                        'pair': pair,
                        'entry_time': entry_data.get('entry_time', trade.open_date_utc),
                        'exit_time': current_time,
                        'entry_price': entry_data.get('entry_price', trade.open_rate),
                        'exit_price': rate,
                        'side': entry_data.get('side', 'long'),
                        'profit_ratio': profit_ratio,
                        'profit_abs': profit_abs,
                        'exit_reason': exit_reason,
                        'entry_buy_signal_count': entry_data.get('buy_signal_count', 0),
                        'entry_sell_signal_count': entry_data.get('sell_signal_count', 0),
                        'entry_rsi_5m': entry_data.get('rsi_5m', 0),
                        'entry_rsi_1h': entry_data.get('rsi_1h', 0),
                        'entry_adx_5m': entry_data.get('adx_5m', 0),
                        'exit_buy_signal_count': exit_indicators['buy_signal_count'],
                        'exit_sell_signal_count': exit_indicators['sell_signal_count'],
                        'exit_rsi_5m': exit_indicators['rsi_5m'],
                        'exit_rsi_1h': exit_indicators['rsi_1h'],
                        'exit_adx_5m': exit_indicators['adx_5m'],
                    }
                    save_trade_to_csv(trade_data)
                
                # Send notification if enabled
                if self.enable_notifications.value:
                    emoji = "🟢" if profit_ratio > 0 else "🔴"
                    message = f"{emoji} EXIT {entry_data.get('side', 'LONG').upper()}\n"
                    message += f"Pair: {pair}\n"
                    message += f"Entry: {entry_data.get('entry_price', trade.open_rate):.8f}\n"
                    message += f"Exit: {rate:.8f}\n"
                    message += f"Profit: {profit_ratio*100:.2f}%\n"
                    message += f"Reason: {exit_reason}\n"
                    message += f"RSI 5m: {exit_indicators['rsi_5m']:.2f}"
                    
                    priority = 5 if profit_ratio > 0 else 3
                    send_ntfy_notification(message, self.ntfy_topic, priority=priority)
                
                # Clean up stored entry data
                if pair in self.entry_indicators:
                    del self.entry_indicators[pair]
        
        except Exception as e:
            logger.error(f"Error in confirm_trade_exit: {e}")
        
        return True
