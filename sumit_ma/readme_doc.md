# Multi-Indicator Freqtrade Strategy

A comprehensive futures trading strategy combining multiple technical indicators across different timeframes.

## 🎯 Features

- **6 Major Indicators**:
  - Sumit MA Signals (18 Moving Averages)
  - Multi SuperTrend (3 configurations)
  - RSI Multi-Timeframe (1m, 5m, 1h)
  - Aroon Multi-Timeframe
  - ADX Multi-Timeframe
  - MACD Multi-Timeframe

- **Advanced Features**:
  - Hyperopt optimization support
  - ntfy.sh notifications (optional)
  - CSV trade logging with full indicator values (optional)
  - Both long and short positions
  - Trailing stop loss

## 📁 File Structure

```
user_data/
├── strategies/
│   ├── MultiIndicatorStrategy.py    # Main strategy file
│   ├── indicators.py                # Custom indicators
│   └── utils.py                     # Notifications & CSV logging
└── logs/
    └── trades_log.csv              # Trade history (auto-created)

config.json                          # Main configuration
hyperopt_config.json                 # Hyperopt settings
```

## 🚀 Installation

### 1. Install Freqtrade

```bash
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
./setup.sh -i
```

### 2. Install Additional Dependencies

```bash
pip install ta
pip install requests
```

### 3. Copy Strategy Files

```bash
# Copy strategy files to user_data/strategies/
cp MultiIndicatorStrategy.py user_data/strategies/
cp indicators.py user_data/strategies/
cp utils.py user_data/strategies/

# Copy config files
cp config.json ./
cp hyperopt_config.json ./
```

### 4. Configure Exchange

Edit `config.json` and add your exchange API credentials:

```json
"exchange": {
    "name": "binance",
    "key": "your-api-key",
    "secret": "your-api-secret",
    ...
}
```

## 🔧 Configuration

### Enable Notifications (Optional)

Edit your config.json or use command line:

```bash
freqtrade trade --strategy MultiIndicatorStrategy \
    --config config.json \
    --enable-notifications true \
    --ntfy-topic "your_unique_topic"
```

**Subscribe to notifications**: Open https://ntfy.sh/your_unique_topic on your phone or browser.

### Enable CSV Logging (Optional)

```bash
freqtrade trade --strategy MultiIndicatorStrategy \
    --config config.json \
    --enable-csv-logging true
```

### Both Features Combined

```bash
freqtrade trade --strategy MultiIndicatorStrategy \
    --config config.json \
    --enable-notifications true \
    --enable-csv-logging true \
    --ntfy-topic "my_trades"
```

## 📊 Usage

### Backtesting

```bash
# Basic backtesting
freqtrade backtesting \
    --strategy MultiIndicatorStrategy \
    --config config.json \
    --timerange 20231101-20240229

# With CSV logging enabled
freqtrade backtesting \
    --strategy MultiIndicatorStrategy \
    --config config.json \
    --timerange 20231101-20240229 \
    --enable-csv-logging true
```

### Hyperopt Optimization

```bash
# Run hyperopt
freqtrade hyperopt \
    --strategy MultiIndicatorStrategy \
    --config config.json \
    --hyperopt-loss SharpeHyperOptLoss \
    --epochs 100 \
    --spaces buy sell

# Show best results
freqtrade hyperopt-show -n -1
```

**Hyperopt Parameters Being Optimized**:
- `buy_ma_cross_threshold`: MA signal threshold for buy (5-15)
- `sell_ma_cross_threshold`: MA signal threshold for sell (5-15)
- `use_supertrend`: Enable/disable SuperTrend filter
- `st_min_aligned`: Minimum aligned SuperTrends (1-3)
- `rsi_5m_buy_threshold`: RSI 5m buy level (25-45)
- `rsi_5m_sell_threshold`: RSI 5m sell level (55-75)
- `rsi_1h_buy_threshold`: RSI 1h buy level (30-50)
- `rsi_1h_sell_threshold`: RSI 1h sell level (50-70)
- `adx_threshold`: Minimum ADX strength (15-30)
- `aroon_osc_buy`: Aroon oscillator buy level (30-70)
- `aroon_osc_sell`: Aroon oscillator sell level (-70 to -30)

### Dry Run

```bash
freqtrade trade \
    --strategy MultiIndicatorStrategy \
    --config config.json \
    --dry-run
```

### Live Trading

```bash
# Make sure to test thoroughly before live trading!
freqtrade trade \
    --strategy MultiIndicatorStrategy \
    --config config.json
```

## 📈 Strategy Logic

### Entry Conditions (LONG)

1. **Sumit MA Signals**: 
   - MA3 crosses above MA11
   - Buy signal count >= threshold

2. **SuperTrend**: Minimum 2 out of 3 bullish (optional)

3. **RSI**: 
   - RSI 5m > 35
   - RSI 1h > 40

4. **ADX**: Strength > 20

5. **Aroon**: Oscillator > 50

6. **MACD**: Histogram > 0

### Entry Conditions (SHORT)

Similar but opposite conditions with sell signals.

### Exit Conditions

- MA signal reversal
- RSI extremes (>75 for long, <25 for short)
- SuperTrend flip
- MACD cross

## 📝 CSV Log Format

The CSV log includes:

**Trade Info**:
- Pair, Entry/Exit times, Prices, Side, Profit, Exit reason

**Entry Indicators**:
- All 6 indicators' values at entry point

**Exit Indicators**:
- All 6 indicators' values at exit point

Example CSV columns:
```
pair, entry_time, exit_time, entry_price, exit_price, side, profit_ratio,
entry_buy_signal_count, entry_rsi_5m, entry_adx_5m, ...
exit_buy_signal_count, exit_rsi_5m, exit_adx_5m, ...
```

## 🔔 Notification Format

Entry notification example:
```
🟢 ENTRY LONG
Pair: BTC/USDT
Price: 45000.00000000
Time: 2024-02-15 10:30:00
Buy Signals: 14
Sell Signals: 4
RSI 5m: 42.50
ADX 5m: 25.30
```

Exit notification example:
```
🟢 EXIT LONG
Pair: BTC/USDT
Entry: 45000.00000000
Exit: 45750.00000000
Profit: 1.67%
Reason: exit_signal
RSI 5m: 72.30
Duration: 0:45:00
```

## 🧪 Testing Notifications

Test your ntfy.sh setup:

```python
from utils import test_ntfy_connection

# Test notification
test_ntfy_connection("your_topic_name")
```

## 📊 Analyze Trade Statistics

```python
from utils import get_trade_statistics

# Get statistics from CSV log
stats = get_trade_statistics("trades_log.csv")
print(f"Win Rate: {stats['win_rate']:.2f}%")
print(f"Total Trades: {stats['total_trades']}")
print(f"Average Profit: {stats['avg_profit_ratio']*100:.2f}%")
```

## ⚙️ Strategy Parameters

```python
timeframe = '5m'
can_short = True
startup_candle_count = 320
stoploss = -0.01
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
max_open_trades = 1
```

## 🐛 Troubleshooting

### Issue: Indicators not calculating

**Solution**: Ensure you have at least 320 candles of data:
```bash
freqtrade download-data \
    --exchange binance \
    --pairs BTC/USDT ETH/USDT \
    --timeframe 5m \
    --days 30
```

### Issue: ntfy notifications not working

**Solution**: Test connection manually:
```bash
curl -d "Test message" https://ntfy.sh/your_topic
```

### Issue: CSV file not created

**Solution**: Check permissions and ensure logs directory exists:
```bash
mkdir -p user_data/logs
chmod 755 user_data/logs
```

## 📚 Additional Resources

- [Freqtrade Documentation](https://www.freqtrade.io/)
- [ntfy.sh Documentation](https://docs.ntfy.sh/)
- [TA-Lib Documentation](https://ta-lib.org/)

## ⚠️ Risk Warning

**This strategy is for educational purposes. Always:**
- Test thoroughly in dry-run mode
- Use proper risk management
- Never invest more than you can afford to lose
- Cryptocurrency trading carries significant risk

## 📄 License

MIT License - Free to use and modify

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Happy Trading! 🚀**