#!/bin/bash

echo "🚀 Setting up Multi-Indicator Freqtrade Strategy"
echo "================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running in freqtrade directory
if [ ! -d "user_data" ]; then
    echo -e "${RED}Error: user_data directory not found!${NC}"
    echo "Please run this script from your freqtrade root directory"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p user_data/strategies
mkdir -p user_data/logs
mkdir -p user_data/data

# Copy strategy files
echo -e "${YELLOW}Copying strategy files...${NC}"
if [ -f "MultiIndicatorStrategy.py" ]; then
    cp MultiIndicatorStrategy.py user_data/strategies/
    cp indicators.py user_data/strategies/
    cp utils.py user_data/strategies/
    echo -e "${GREEN}✓ Strategy files copied${NC}"
else
    echo -e "${RED}Error: Strategy files not found in current directory${NC}"
    exit 1
fi

# Install additional dependencies
echo -e "${YELLOW}Installing additional dependencies...${NC}"
pip install ta requests

# Download sample data (optional)
read -p "Download sample data for BTC/USDT and ETH/USDT? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Downloading data...${NC}"
    freqtrade download-data \
        --exchange binance \
        --pairs BTC/USDT ETH/USDT \
        --timeframe 5m \
        --days 30 \
        --trading-mode futures
    echo -e "${GREEN}✓ Data downloaded${NC}"
fi

# Test notification setup (optional)
read -p "Set up ntfy.sh notifications? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your ntfy.sh topic name: " NTFY_TOPIC
    
    # Test notification
    curl -d "🎉 Freqtrade setup complete! Notifications are working." \
         -H "Title: Setup Complete" \
         -H "Priority: 4" \
         "https://ntfy.sh/$NTFY_TOPIC"
    
    echo -e "${GREEN}✓ Test notification sent to topic: $NTFY_TOPIC${NC}"
    echo -e "${YELLOW}Subscribe at: https://ntfy.sh/$NTFY_TOPIC${NC}"
fi

# Create default config if not exists
if [ ! -f "config.json" ]; then
    echo -e "${YELLOW}Creating default config.json...${NC}"
    cp config.json.example config.json 2>/dev/null || {
        echo -e "${YELLOW}Please configure config.json with your exchange API keys${NC}"
    }
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo "1. Edit config.json with your exchange API credentials"
echo "2. Run backtesting:"
echo "   freqtrade backtesting --strategy MultiIndicatorStrategy --config config.json --timerange 20240101-20240201"
echo ""
echo "3. Run hyperopt optimization:"
echo "   freqtrade hyperopt --strategy MultiIndicatorStrategy --config config.json --epochs 100 --spaces buy sell"
echo ""
echo "4. Start dry-run trading:"
echo "   freqtrade trade --strategy MultiIndicatorStrategy --config config.json --dry-run"
echo ""
echo "For more information, see README.md"
echo ""