# Quick Reference Guide

## 🚀 Quick Start Commands

### 1. Installation
```bash
# Run setup script
chmod +x setup.sh
./setup.sh

# Or manual installation
pip install ta requests
mkdir -p user_data/strategies user_data/logs
```

### 2. Download Data
```bash
freqtrade download-data \
    --exchange binance \
    --pairs BTC/USDT ETH/USDT \
    --timeframe 5m \
    --days 30 \
    --trading-mode futures
```

### 3. Backtesting
```bash
# Basic backtest
freqtrade backtesting \
    --strategy MultiIndicatorStrategy \
    --config config.json \
    --timerange 20240101-20240201

# With CSV logging
freqtrade backtesting \
    --strategy MultiIndicatorStrategy \
    --config config.json \
    --timerange 20240101-20240201 \
    --enable-csv-logging true
```

### 4. Hyperopt
```bash
# Optimize buy/sell parameters
freqtrade hyperopt \
    --strategy MultiIndicatorStrategy \
    --config config.json \
    --hyperopt-loss SharpeHyperOptLoss \
    --epochs 100 \
    --spaces buy sell

# Show best results
freqtrade hyperopt-show -n -1
```

### 5. Dry Run
```bash
freqtrade trade \
    --strategy MultiIndicatorStrategy \
    --config config.json \
    --dry-run
```

### 6. Live Trading
```bash
freqtrade trade \
    --strategy MultiIndicatorStrategy \
    --config config.json
```

## 🔔 Notification Setup

### Test ntfy.sh
```bash
# Send test message
curl -d "Test message" https://ntfy.sh/your_topic_name

# Subscribe on phone
# Install ntfy app or visit: https://ntfy.sh/your_topic_name
```

### Enable in Freqtrade
Add to config.json:
```json
{
    "enable_notifications": true,
    "ntfy_topic": "your_unique_topic"
}
```

## 📊 CSV Analysis

### Run Analysis Script
```bash
python analyze_trades.py
```

### Manual Analysis
```python
import pandas as pd

# Load trades
df = pd.read_csv('user_data/logs/trades_log.csv')

# Basic stats
print(f"Total trades: {len(df)}")
print(f"Win rate: {(df['profit_ratio'] > 0).sum() / len(df) * 100:.2f}%")
print(f"Total profit: {df['profit_ratio'].sum() * 100:.2f}%")
```

## 📝 Strategy Parameters

### Core Settings
```python
timeframe = '5m'              # Trading timeframe
can_short = True              # Enable short positions
stoploss = -0.01              # Stop loss (1%)
max_open_trades = 1           # Maximum positions
```

### Trailing Stop
```python
trailing_stop = True
trailing_stop_positive = 0.005         # Start trailing at 0.5%
trailing_stop_positive_offset = 0.01   # Offset by 1%
trailing_only_offset_is_reached = True
```

### Hyperopt Parameters (Optimizable)
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `buy_ma_cross_threshold` | 5-15 | 10 | MA signals for buy entry |
| `sell_ma_cross_threshold` | 5-15 | 10 | MA signals for sell entry |
| `use_supertrend` | True/False | True | Enable SuperTrend filter |
| `st_min_aligned` | 1-3 | 2 | Min aligned SuperTrends |
| `rsi_5m_buy_threshold` | 25-45 | 35 | RSI 5m buy level |
| `rsi_5m_sell_threshold` | 55-75 | 65 | RSI 5m sell level |
| `rsi_1h_buy_threshold` | 30-50 | 40 | RSI 1h buy level |
| `rsi_1h_sell_threshold` | 50-70 | 60 | RSI 1h sell level |
| `adx_threshold` | 15-30 | 20 | Minimum ADX strength |
| `aroon_osc_buy` | 30-70 | 50 | Aroon osc buy level |
| `aroon_osc_sell` | -70--30 | -50 | Aroon osc sell level |

## 🎯 Indicator Summary

### 1. Sumit MA Signals
- **18 Moving Averages** on OHLC/4
- Periods: 3, 9, 15, 21, 27, 31, 37, 51, 65, 81, 101, 121, 131, 151, 171, 201, 251, 301
- **Buy/Sell Counts**: Price position vs MAs
- **MA3 & MA11**: Signal smoothing

### 2. Multi SuperTrend
- **ST 7,3**: Fast SuperTrend
- **ST 10,2**: Medium SuperTrend
- **ST 21,7**: Slow SuperTrend
- **Direction**: 1 (bullish) or -1 (bearish)
- **Flips**: Trend changes

### 3. RSI Multi-Timeframe
- **1min, 5min, 1hr** timeframes
- Calculated on average price: (O+H+L+C)/4
- Period: 14
- SMA smoothing: 7

### 4. Aroon Indicator
- **Aroon Up/Down**: Trend strength
- **Aroon Oscillator**: Up - Down
- Period: 14

### 5. ADX (Average Directional Index)
- **ADX**: Trend strength (>20 = strong)
- **+DI/-DI**: Directional indicators
- Period: 14

### 6. MACD
- **Fast**: 12, **Slow**: 26, **Signal**: 9
- **Histogram**: MACD - Signal
- **Cross**: Entry/exit signals

## 🔧 Troubleshooting

### Problem: Not enough candles
```bash
# Solution: Increase startup_candle_count or download more data
freqtrade download-data --days 60
```

### Problem: Indicators showing NaN
```bash
# Solution: Check data quality
freqtrade show-trades --db-url sqlite:///tradesv3.sqlite
```

### Problem: Hyperopt too slow
```bash
# Solution: Reduce epochs or use parallel jobs
freqtrade hyperopt --epochs 50 --jobs 4
```

### Problem: Notifications not received
```bash
# Solution: Test ntfy connection
curl -d "Test" https://ntfy.sh/your_topic

# Check logs
tail -f user_data/logs/freqtrade.log
```

## 📈 Performance Tips

1. **Optimize regularly**: Market conditions change
2. **Use multiple timeframes**: Confirm trends
3. **Monitor win rate**: Should be >50% ideally
4. **Track indicators**: Analyze what works
5. **Adjust parameters**: Based on CSV analysis
6. **Risk management**: Never risk >1-2% per trade

## 🔒 Security Checklist

- [ ] Never share API keys
- [ ] Use IP whitelist on exchange
- [ ] Enable 2FA on exchange
- [ ] Keep API keys in secure config
- [ ] Use read-only keys for backtesting
- [ ] Test thoroughly in dry-run first
- [ ] Monitor positions regularly

## 📚 Useful Links

- [Freqtrade Docs](https://www.freqtrade.io/en/stable/)
- [Strategy Callbacks](https://www.freqtrade.io/en/stable/strategy-callbacks/)
- [Hyperopt Guide](https://www.freqtrade.io/en/stable/hyperopt/)
- [ntfy.sh Docs](https://docs.ntfy.sh/)

## 🆘 Support

If you encounter issues:

1. Check logs: `tail -f user_data/logs/freqtrade.log`
2. Verify config: `freqtrade show-config`
3. Test strategy: `freqtrade test-strategy`
4. Join Freqtrade Discord for help

---

**Remember**: Always test in dry-run mode before live trading!