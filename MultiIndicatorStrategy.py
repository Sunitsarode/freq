"""
Multi-Indicator Strategy for Freqtrade
Combines Sumit MA Signals, SuperTrend, RSI, Aroon, ADX, and MACD across multiple timeframes
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

logger = logging.getLogger(__name__)


class MultiIndicatorStrategy(IStrategy):
    """
    Advanced multi-indicator strategy with comprehensive signal generation
    """
    
    # Strategy settings
    timeframe = '5m'
    can_short = True
    startup_candle_count = 320  # Increased for MA301 + safety margin
    process_only_new_candles = True
    
    # Risk management
    stoploss = -0.01
    
    # Trading mode
    trading_mode = 'futures'
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True
    
    # Exit settings
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    
    # Position management
    max_open_trades = 1
    
    # Optional features (can be enabled via config)
    enable_notifications = BooleanParameter(default=False, space='buy', optimize=False)
    enable_csv_logging = BooleanParameter(default=False, space='buy', optimize=False)
    ntfy_topic = 'freqtrade_alerts'  # Default ntfy.sh topic
    
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
    
    # Store for entry indicators
    entry_indicators = {}
    
    # Plot configuration for freqtrade-plot
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
        """
        Define additional pairs/timeframes if needed
        """
        return []
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all indicators
        """
        if len(dataframe) < self.startup_candle_count:
            logger.warning(f"Not enough data: {len(dataframe)} < {self.startup_candle_count}")
            return dataframe
        
        # Make a copy to avoid issues
        df = dataframe.copy()
        
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
        
        # Merge 1h RSI back to 5m timeframe
        if len(rsi_data['1h']) > 0:
            # Forward fill 1h data to match 5m timeframe
            df['rsi_1h'] = rsi_data['1h']['rsi'].reindex(df.index, method='ffill')
            df['rsi_1h_sma'] = rsi_data['1h']['rsi_sma'].reindex(df.index, method='ffill')
        else:
            df['rsi_1h'] = df['rsi_5m']
            df['rsi_1h_sma'] = df['rsi_5m_sma']
        
        # Fill any remaining NaNs with forward fill
        df['rsi_1h'] = df['rsi_1h'].fillna(method='ffill').fillna(50)
        df['rsi_1h_sma'] = df['rsi_1h_sma'].fillna(method='ffill').fillna(50)
        
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
        
        # Fill any NaN values in critical columns to prevent comparison errors
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
        
        return df
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions for long and short positions
        """
        # Initialize entry columns with integer type
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # LONG ENTRY CONDITIONS
        # Build each condition separately to avoid dtype issues
        
        # 1. MA Signal cross
        ma_cross_long = (
            (dataframe['buy_signal_ma3'] > dataframe['buy_signal_ma11']) &
            (dataframe['buy_signal_ma3'].shift(1) <= dataframe['buy_signal_ma11'].shift(1)) &
            (dataframe['buy_signal_count'] >= self.buy_ma_cross_threshold.value)
        )
        
        # 2. SuperTrend alignment
        if self.use_supertrend.value:
            st_bullish = (
                (dataframe['st_7_3_direction'] == 1) &
                (dataframe['st_10_2_direction'] == 1)
            )
        else:
            st_bullish = True
        
        # 3. RSI conditions
        rsi_long = (
            (dataframe['rsi_5m'] > self.rsi_5m_buy_threshold.value) &
            (dataframe['rsi_1h'] > self.rsi_1h_buy_threshold.value)
        )
        
        # 4. ADX strength
        adx_strong = (dataframe['adx_5m'] > self.adx_threshold.value)
        
        # 5. Aroon bullish
        aroon_long = (dataframe['aroon_osc_5m'] > self.aroon_osc_buy.value)
        
        # 6. MACD positive
        macd_long = (dataframe['macd_hist_5m'] > 0)
        
        # Combine all conditions
        long_entry = ma_cross_long & st_bullish & rsi_long & adx_strong & aroon_long & macd_long
        
        dataframe.loc[long_entry, 'enter_long'] = 1
        
        # SHORT ENTRY CONDITIONS (if enabled)
        if self.can_short:
            # 1. MA Signal cross
            ma_cross_short = (
                (dataframe['sell_signal_ma3'] > dataframe['sell_signal_ma11']) &
                (dataframe['sell_signal_ma3'].shift(1) <= dataframe['sell_signal_ma11'].shift(1)) &
                (dataframe['sell_signal_count'] >= self.sell_ma_cross_threshold.value)
            )
            
            # 2. SuperTrend alignment
            if self.use_supertrend.value:
                st_bearish = (
                    (dataframe['st_7_3_direction'] == -1) &
                    (dataframe['st_10_2_direction'] == -1)
                )
            else:
                st_bearish = True
            
            # 3. RSI conditions
            rsi_short = (
                (dataframe['rsi_5m'] < self.rsi_5m_sell_threshold.value) &
                (dataframe['rsi_1h'] < self.rsi_1h_sell_threshold.value)
            )
            
            # 4. ADX strength
            adx_strong_short = (dataframe['adx_5m'] > self.adx_threshold.value)
            
            # 5. Aroon bearish
            aroon_short = (dataframe['aroon_osc_5m'] < self.aroon_osc_sell.value)
            
            # 6. MACD negative
            macd_short = (dataframe['macd_hist_5m'] < 0)
            
            # Combine all conditions
            short_entry = ma_cross_short & st_bearish & rsi_short & adx_strong_short & aroon_short & macd_short
            
            dataframe.loc[short_entry, 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions
        """
        # Initialize exit columns with integer type
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # EXIT LONG - separate conditions
        
        # MA cross reversal
        ma_reversal_long = (
            (dataframe['sell_signal_ma3'] > dataframe['buy_signal_ma3']) &
            (dataframe['sell_signal_ma3'].shift(1) <= dataframe['buy_signal_ma3'].shift(1))
        )
        
        # RSI overbought
        rsi_overbought = (dataframe['rsi_5m'] > 75)
        
        # MACD cross down
        macd_cross_down = (
            (dataframe['macd_5m'] < dataframe['macd_signal_5m']) &
            (dataframe['macd_5m'].shift(1) >= dataframe['macd_signal_5m'].shift(1))
        )
        
        # Combine exit conditions for long
        exit_long = ma_reversal_long | rsi_overbought | macd_cross_down
        
        dataframe.loc[exit_long, 'exit_long'] = 1
        
        # EXIT SHORT (if enabled)
        if self.can_short:
            # MA cross reversal
            ma_reversal_short = (
                (dataframe['buy_signal_ma3'] > dataframe['sell_signal_ma3']) &
                (dataframe['buy_signal_ma3'].shift(1) <= dataframe['sell_signal_ma3'].shift(1))
            )
            
            # RSI oversold
            rsi_oversold = (dataframe['rsi_5m'] < 25)
            
            # MACD cross up
            macd_cross_up = (
                (dataframe['macd_5m'] > dataframe['macd_signal_5m']) &
                (dataframe['macd_5m'].shift(1) <= dataframe['macd_signal_5m'].shift(1))
            )
            
            # Combine exit conditions for short
            exit_short = ma_reversal_short | rsi_oversold | macd_cross_up
            
            dataframe.loc[exit_short, 'exit_short'] = 1
        
        return dataframe
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Called right before placing an entry order.
        Store entry indicators for later use.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            
            # Store entry indicators
            self.entry_indicators[pair] = {
                'entry_time': current_time,
                'entry_price': rate,
                'side': side,
                'buy_signal_count': last_candle.get('buy_signal_count', 0),
                'sell_signal_count': last_candle.get('sell_signal_count', 0),
                'buy_signal_ma3': last_candle.get('buy_signal_ma3', 0),
                'buy_signal_ma11': last_candle.get('buy_signal_ma11', 0),
                'sell_signal_ma3': last_candle.get('sell_signal_ma3', 0),
                'sell_signal_ma11': last_candle.get('sell_signal_ma11', 0),
                'st_7_3_direction': last_candle.get('st_7_3_direction', 0),
                'st_10_2_direction': last_candle.get('st_10_2_direction', 0),
                'st_21_7_direction': last_candle.get('st_21_7_direction', 0),
                'st_avg': last_candle.get('st_avg', 0),
                'rsi_5m': last_candle.get('rsi_5m', 0),
                'rsi_1h': last_candle.get('rsi_1h', 0),
                'aroon_osc_5m': last_candle.get('aroon_osc_5m', 0),
                'adx_5m': last_candle.get('adx_5m', 0),
                'macd_5m': last_candle.get('macd_5m', 0),
                'macd_hist_5m': last_candle.get('macd_hist_5m', 0),
            }
            
            # Send notification if enabled
            if self.enable_notifications.value:
                message = f"🟢 ENTRY {side.upper()}\n"
                message += f"Pair: {pair}\n"
                message += f"Price: {rate:.8f}\n"
                message += f"Time: {current_time}\n"
                message += f"Buy Signals: {last_candle.get('buy_signal_count', 0)}\n"
                message += f"Sell Signals: {last_candle.get('sell_signal_count', 0)}\n"
                message += f"RSI 5m: {last_candle.get('rsi_5m', 0):.2f}\n"
                message += f"ADX 5m: {last_candle.get('adx_5m', 0):.2f}"
                
                send_ntfy_notification(message, self.ntfy_topic, priority=4)
        
        return True
    
    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing an exit order.
        Log exit indicators and send notifications/save to CSV.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            
            # Get entry indicators
            entry_data = self.entry_indicators.get(pair, {})
            
            # Calculate profit/loss
            profit_ratio = trade.calc_profit_ratio(rate)
            profit_abs = trade.calc_profit(rate)
            
            # Prepare trade data
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
                
                # Entry indicators
                'entry_buy_signal_count': entry_data.get('buy_signal_count', 0),
                'entry_sell_signal_count': entry_data.get('sell_signal_count', 0),
                'entry_buy_signal_ma3': entry_data.get('buy_signal_ma3', 0),
                'entry_buy_signal_ma11': entry_data.get('buy_signal_ma11', 0),
                'entry_st_7_3_direction': entry_data.get('st_7_3_direction', 0),
                'entry_st_10_2_direction': entry_data.get('st_10_2_direction', 0),
                'entry_st_21_7_direction': entry_data.get('st_21_7_direction', 0),
                'entry_rsi_5m': entry_data.get('rsi_5m', 0),
                'entry_rsi_1h': entry_data.get('rsi_1h', 0),
                'entry_adx_5m': entry_data.get('adx_5m', 0),
                'entry_macd_hist_5m': entry_data.get('macd_hist_5m', 0),
                
                # Exit indicators
                'exit_buy_signal_count': last_candle.get('buy_signal_count', 0),
                'exit_sell_signal_count': last_candle.get('sell_signal_count', 0),
                'exit_buy_signal_ma3': last_candle.get('buy_signal_ma3', 0),
                'exit_buy_signal_ma11': last_candle.get('buy_signal_ma11', 0),
                'exit_st_7_3_direction': last_candle.get('st_7_3_direction', 0),
                'exit_st_10_2_direction': last_candle.get('st_10_2_direction', 0),
                'exit_st_21_7_direction': last_candle.get('st_21_7_direction', 0),
                'exit_rsi_5m': last_candle.get('rsi_5m', 0),
                'exit_rsi_1h': last_candle.get('rsi_1h', 0),
                'exit_adx_5m': last_candle.get('adx_5m', 0),
                'exit_macd_hist_5m': last_candle.get('macd_hist_5m', 0),
            }
            
            # Send notification if enabled
            if self.enable_notifications.value:
                emoji = "🟢" if profit_ratio > 0 else "🔴"
                message = f"{emoji} EXIT {entry_data.get('side', 'LONG').upper()}\n"
                message += f"Pair: {pair}\n"
                message += f"Entry: {trade_data['entry_price']:.8f}\n"
                message += f"Exit: {rate:.8f}\n"
                message += f"Profit: {profit_ratio*100:.2f}%\n"
                message += f"Reason: {exit_reason}\n"
                message += f"RSI 5m: {last_candle.get('rsi_5m', 0):.2f}\n"
                message += f"Duration: {current_time - trade_data['entry_time']}"
                
                priority = 5 if profit_ratio > 0 else 3
                send_ntfy_notification(message, self.ntfy_topic, priority=priority)
            
            # Save to CSV if enabled
            if self.enable_csv_logging.value:
                save_trade_to_csv(trade_data)
            
            # Clean up stored entry data
            if pair in self.entry_indicators:
                del self.entry_indicators[pair]
        
        return True
