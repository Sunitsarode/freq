# -*- coding: utf-8 -*-
"""
Backtesting Logger - Dynamic indicator support
Accepts any indicators passed from strategy
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BacktestLogger:
    """
    Logs backtesting trades to CSV with dynamic indicator columns
    """
    
    def __init__(self, strategy_name: str = "Strategy"):
        self.strategy_name = strategy_name
        self.csv_file = None
        self.csv_writer = None
        self.file_handle = None
        self.trades_in_progress = {}
        self.fieldnames = None
        
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with timestamp"""
        try:
            output_dir = Path('user_data/strategies/sumit_ma/backtests')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.strategy_name}_{timestamp}.csv"
            self.csv_file = output_dir / filename
            
            self.file_handle = open(self.csv_file, 'w', newline='')
            
            self.fieldnames = [
                'pair', 'entry_time', 'exit_time', 'duration_minutes',
                'side', 'entry_price', 'exit_price', 'profit_pct', 'exit_reason',
                'entry_score', 'exit_score'
            ]
            
            self.csv_writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames, extrasaction='ignore')
            self.csv_writer.writeheader()
            
            logger.info(f">> Backtest CSV: {self.csv_file}")
            
        except Exception as e:
            logger.error(f">> Failed to initialize backtest logger: {e}")
            self.csv_writer = None
    
    def _update_fieldnames(self, indicators: Dict):
        """Dynamically update CSV columns based on indicators passed"""
        if not self.csv_writer:
            return
            
        new_fields = []
        for key in indicators.keys():
            entry_field = f'entry_{key}'
            exit_field = f'exit_{key}'
            if entry_field not in self.fieldnames:
                new_fields.append(entry_field)
            if exit_field not in self.fieldnames:
                new_fields.append(exit_field)
        
        if new_fields:
            self.fieldnames.extend(new_fields)
            self.file_handle.close()
            
            temp_data = []
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                temp_data = list(reader)
            
            self.file_handle = open(self.csv_file, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames, extrasaction='ignore')
            self.csv_writer.writeheader()
            
            for row in temp_data:
                self.csv_writer.writerow(row)
    
    def log_entry(self, pair: str, entry_time: datetime, entry_price: float, 
                  side: str, indicators: Dict):
        """Store entry information"""
        try:
            self._update_fieldnames(indicators)
            
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
        """Complete trade log with exit data"""
        if not self.csv_writer:
            return
        
        try:
            entry_data = self.trades_in_progress.get(pair)
            if not entry_data:
                logger.warning(f"No entry data for {pair}")
                return
            
            duration = (exit_time - entry_data['entry_time']).total_seconds() / 60
            
            row = {
                'pair': pair,
                'entry_time': entry_data['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_minutes': f"{duration:.2f}",
                'side': entry_data['side'],
                'entry_price': f"{entry_data['entry_price']:.8f}",
                'exit_price': f"{exit_price:.8f}",
                'profit_pct': f"{profit_ratio * 100:.2f}",
                'exit_reason': exit_reason,
                'entry_score': entry_data['entry_indicators'].get('entry_score', 0),
                'exit_score': indicators.get('exit_score', 0),
            }
            
            for key, value in entry_data['entry_indicators'].items():
                if key != 'entry_score':
                    row[f'entry_{key}'] = f"{value:.2f}" if isinstance(value, float) else value
            
            for key, value in indicators.items():
                if key != 'exit_score':
                    row[f'exit_{key}'] = f"{value:.2f}" if isinstance(value, float) else value
            
            self.csv_writer.writerow(row)
            self.file_handle.flush()
            
            del self.trades_in_progress[pair]
            
        except Exception as e:
            logger.error(f"Error logging exit for {pair}: {e}")
    
    def close(self):
        """Close CSV file"""
        try:
            if self.file_handle:
                self.file_handle.close()
                logger.info(f">> Backtest CSV saved: {self.csv_file}")
        except Exception as e:
            logger.error(f"Error closing logger: {e}")
    
    def __del__(self):
        self.close()
