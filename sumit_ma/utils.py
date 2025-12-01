"""
Utility functions for notifications and CSV logging
"""

import csv
import os
import requests
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def send_ntfy_notification(message: str, topic: str, priority: int = 3, 
                           title: Optional[str] = None) -> bool:
    """
    Send notification to ntfy.sh
    
    Args:
        message: Message body
        topic: ntfy.sh topic name
        priority: 1-5 (1=min, 3=default, 5=max)
        title: Optional title for notification
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        url = f"https://ntfy.sh/{topic}"
        
        headers = {
            "Priority": str(priority),
            "Tags": "chart_with_upwards_trend"
        }
        
        if title:
            headers["Title"] = title
        else:
            headers["Title"] = "Freqtrade Alert"
        
        response = requests.post(
            url,
            data=message.encode('utf-8'),
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Notification sent successfully to topic: {topic}")
            return True
        else:
            logger.error(f"Failed to send notification. Status code: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        return False


def save_trade_to_csv(trade_data: Dict, filename: str = "trades_log.csv") -> bool:
    """
    Save trade information to CSV file
    
    Args:
        trade_data: Dictionary containing trade information
        filename: CSV filename (default: trades_log.csv)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create user_data/logs directory if it doesn't exist
        log_dir = Path("user_data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = log_dir / filename
        file_exists = filepath.exists()
        
        # Define CSV columns
        fieldnames = [
            'pair', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'side', 'profit_ratio', 'profit_abs', 'exit_reason',
            
            # Entry indicators
            'entry_buy_signal_count', 'entry_sell_signal_count',
            'entry_buy_signal_ma3', 'entry_buy_signal_ma11',
            'entry_st_7_3_direction', 'entry_st_10_2_direction', 'entry_st_21_7_direction',
            'entry_rsi_5m', 'entry_rsi_1h',
            'entry_adx_5m', 'entry_macd_hist_5m',
            
            # Exit indicators
            'exit_buy_signal_count', 'exit_sell_signal_count',
            'exit_buy_signal_ma3', 'exit_buy_signal_ma11',
            'exit_st_7_3_direction', 'exit_st_10_2_direction', 'exit_st_21_7_direction',
            'exit_rsi_5m', 'exit_rsi_1h',
            'exit_adx_5m', 'exit_macd_hist_5m'
        ]
        
        with open(filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Format datetime objects
            formatted_data = trade_data.copy()
            for key in ['entry_time', 'exit_time']:
                if key in formatted_data and formatted_data[key]:
                    if isinstance(formatted_data[key], datetime):
                        formatted_data[key] = formatted_data[key].strftime('%Y-%m-%d %H:%M:%S')
            
            # Write trade data
            writer.writerow(formatted_data)
        
        logger.info(f"Trade data saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving trade to CSV: {e}")
        return False


def load_trades_from_csv(filename: str = "trades_log.csv") -> list:
    """
    Load trades from CSV file for analysis
    
    Args:
        filename: CSV filename
    
    Returns:
        list: List of trade dictionaries
    """
    try:
        filepath = Path("user_data/logs") / filename
        
        if not filepath.exists():
            logger.warning(f"CSV file not found: {filepath}")
            return []
        
        trades = []
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                trades.append(row)
        
        logger.info(f"Loaded {len(trades)} trades from {filepath}")
        return trades
        
    except Exception as e:
        logger.error(f"Error loading trades from CSV: {e}")
        return []


def get_trade_statistics(filename: str = "trades_log.csv") -> Dict:
    """
    Calculate statistics from logged trades
    
    Args:
        filename: CSV filename
    
    Returns:
        dict: Trade statistics
    """
    trades = load_trades_from_csv(filename)
    
    if not trades:
        return {}
    
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if float(t.get('profit_ratio', 0)) > 0)
    losing_trades = total_trades - winning_trades
    
    total_profit = sum(float(t.get('profit_ratio', 0)) for t in trades)
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    stats = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_profit_ratio': total_profit,
        'avg_profit_ratio': avg_profit,
    }
    
    return stats


def test_ntfy_connection(topic: str = "freqtrade_test") -> bool:
    """
    Test ntfy.sh connection
    
    Args:
        topic: Test topic name
    
    Returns:
        bool: True if connection successful
    """
    test_message = f"🧪 Freqtrade notification test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return send_ntfy_notification(test_message, topic, priority=1, title="Connection Test")
