"""
Backtesting Logger for Multi-Indicator Strategy
Logs detailed trade information with entry/exit indicators to CSV
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BacktestLogger:
    """
    Logs backtesting trades to CSV with complete indicator data
    CSV Format: <StrategyName>_YYYYMMDD_HHMMSS.csv
    """
    
    def __init__(self, strategy_name: str = "MultiIndicatorStrategy"):
        """
        Initialize logger with timestamped CSV file
        
        Args:
            strategy_name: Name of the strategy (used in filename)
        """
        self.strategy_name = strategy_name
        self.csv_file = None
        self.csv_writer = None
        self.file_handle = None
        self.trades_in_progress = {}  # Store entry data temporarily
        
        # Initialize CSV file
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with timestamp in sumit_ma/backtests/ folder"""
        try:
            # Create directory if not exists
            output_dir = Path('user_data/strategies/sumit_ma/backtests')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.strategy_name}_{timestamp}.csv"
            self.csv_file = output_dir / filename
            
            # Open file and create CSV writer
            self.file_handle = open(self.csv_file, 'w', newline='')
            
            # Define column headers (grouped by indicator for easy analysis)
            fieldnames = [
                # Basic info
                'pair', 'entry_time', 'exit_time', 'duration_minutes', 
                'side', 'entry_price', 'exit_price', 'profit_pct', 'exit_reason',
                
                # Indicators - grouped entry/exit pairs
                'entry_buy_signal_count', 'exit_buy_signal_count',
                'entry_sell_signal_count', 'exit_sell_signal_count',
                'entry_rsi_5m', 'exit_rsi_5m',
                'entry_rsi_1h', 'exit_rsi_1h',
                'entry_adx_5m', 'exit_adx_5m',
                'entry_st_7_3_direction', 'exit_st_7_3_direction',
                'entry_st_10_2_direction', 'exit_st_10_2_direction',
                'entry_st_21_7_direction', 'exit_st_21_7_direction',
                'entry_aroon_osc_5m', 'exit_aroon_osc_5m',
                'entry_macd_hist_5m', 'exit_macd_hist_5m',
            ]
            
            self.csv_writer = csv.DictWriter(self.file_handle, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            
            logger.info(f"✅ Backtest logger initialized: {self.csv_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize backtest logger: {e}")
            self.csv_writer = None
    
    def log_entry(self, pair: str, entry_time: datetime, entry_price: float, 
                  side: str, indicators: Dict):
        """
        Store entry information temporarily
        
        Args:
            pair: Trading pair
            entry_time: Entry timestamp
            entry_price: Entry price
            side: 'long' or 'short'
            indicators: Dictionary of indicator values at entry
        """
        try:
            self.trades_in_progress[pair] = {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'side': side,
                'entry_indicators': indicators
            }
            
        except Exception as e:
            logger.error(f"Error logging entry for {pair}: {e}")
    
    def log_exit(self, pair: str, exit_time: datetime, exit_price: float,
                 profit_ratio: float, exit_reason: str, indicators: Dict):
        """
        Complete trade log with exit data and write to CSV
        
        Args:
            pair: Trading pair
            exit_time: Exit timestamp
            exit_price: Exit price
            profit_ratio: Profit as ratio (0.02 = 2%)
            exit_reason: Exit reason
            indicators: Dictionary of indicator values at exit
        """
        if not self.csv_writer:
            return
        
        try:
            # Get entry data
            entry_data = self.trades_in_progress.get(pair)
            if not entry_data:
                logger.warning(f"No entry data found for {pair}, skipping exit log")
                return
            
            # Calculate duration
            duration = (exit_time - entry_data['entry_time']).total_seconds() / 60
            
            # Prepare row data
            row = {
                # Basic info
                'pair': pair,
                'entry_time': entry_data['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_minutes': f"{duration:.2f}",
                'side': entry_data['side'],
                'entry_price': f"{entry_data['entry_price']:.8f}",
                'exit_price': f"{exit_price:.8f}",
                'profit_pct': f"{profit_ratio * 100:.2f}",
                'exit_reason': exit_reason,
                
                # Entry indicators
                'entry_buy_signal_count': entry_data['entry_indicators'].get('buy_signal_count', 0),
                'entry_sell_signal_count': entry_data['entry_indicators'].get('sell_signal_count', 0),
                'entry_rsi_5m': f"{entry_data['entry_indicators'].get('rsi_5m', 0):.2f}",
                'entry_rsi_1h': f"{entry_data['entry_indicators'].get('rsi_1h', 0):.2f}",
                'entry_adx_5m': f"{entry_data['entry_indicators'].get('adx_5m', 0):.2f}",
                'entry_st_7_3_direction': entry_data['entry_indicators'].get('st_7_3_direction', 0),
                'entry_st_10_2_direction': entry_data['entry_indicators'].get('st_10_2_direction', 0),
                'entry_st_21_7_direction': entry_data['entry_indicators'].get('st_21_7_direction', 0),
                'entry_aroon_osc_5m': f"{entry_data['entry_indicators'].get('aroon_osc_5m', 0):.2f}",
                'entry_macd_hist_5m': f"{entry_data['entry_indicators'].get('macd_hist_5m', 0):.2f}",
                
                # Exit indicators
                'exit_buy_signal_count': indicators.get('buy_signal_count', 0),
                'exit_sell_signal_count': indicators.get('sell_signal_count', 0),
                'exit_rsi_5m': f"{indicators.get('rsi_5m', 0):.2f}",
                'exit_rsi_1h': f"{indicators.get('rsi_1h', 0):.2f}",
                'exit_adx_5m': f"{indicators.get('adx_5m', 0):.2f}",
                'exit_st_7_3_direction': indicators.get('st_7_3_direction', 0),
                'exit_st_10_2_direction': indicators.get('st_10_2_direction', 0),
                'exit_st_21_7_direction': indicators.get('st_21_7_direction', 0),
                'exit_aroon_osc_5m': f"{indicators.get('aroon_osc_5m', 0):.2f}",
                'exit_macd_hist_5m': f"{indicators.get('macd_hist_5m', 0):.2f}",
            }
            
            # Write to CSV
            self.csv_writer.writerow(row)
            self.file_handle.flush()  # Ensure data is written immediately
            
            # Clean up entry data
            del self.trades_in_progress[pair]
            
            logger.debug(f"✅ Trade logged: {pair} | Profit: {profit_ratio*100:.2f}%")
            
        except Exception as e:
            logger.error(f"Error logging exit for {pair}: {e}")
    
    def close(self):
        """Close CSV file"""
        try:
            if self.file_handle:
                self.file_handle.close()
                logger.info(f"✅ Backtest log saved: {self.csv_file}")
        except Exception as e:
            logger.error(f"Error closing backtest logger: {e}")
    
    def __del__(self):
        """Ensure file is closed on deletion"""
        self.close()
