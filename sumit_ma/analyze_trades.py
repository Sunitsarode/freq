# -*- coding: utf-8 -*-
"""
Trade Analysis Script
Analyzes trades from CSV log file and generates insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_trades(filename='trades_log.csv'):
    """Load trades from CSV"""
    filepath = Path('user_data/logs') / filename
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    # Convert datetime columns
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['duration'] = df['exit_time'] - df['entry_time']
    
    return df


def calculate_statistics(df):
    """Calculate comprehensive trading statistics"""
    
    if df is None or len(df) == 0:
        print("No trades to analyze")
        return
    
    total_trades = len(df)
    winning_trades = len(df[df['profit_ratio'] > 0])
    losing_trades = len(df[df['profit_ratio'] <= 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_profit = df['profit_ratio'].sum()
    avg_profit = df['profit_ratio'].mean()
    
    avg_win = df[df['profit_ratio'] > 0]['profit_ratio'].mean() if winning_trades > 0 else 0
    avg_loss = df[df['profit_ratio'] <= 0]['profit_ratio'].mean() if losing_trades > 0 else 0
    
    best_trade = df.loc[df['profit_ratio'].idxmax()] if total_trades > 0 else None
    worst_trade = df.loc[df['profit_ratio'].idxmin()] if total_trades > 0 else None
    
    # Duration stats
    avg_duration = df['duration'].mean()
    
    # Long vs Short performance
    long_trades = df[df['side'] == 'long']
    short_trades = df[df['side'] == 'short']
    
    stats = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_profit_pct': total_profit * 100,
        'avg_profit_pct': avg_profit * 100,
        'avg_win_pct': avg_win * 100,
        'avg_loss_pct': avg_loss * 100,
        'avg_duration': avg_duration,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_win_rate': (len(long_trades[long_trades['profit_ratio'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0,
        'short_win_rate': (len(short_trades[short_trades['profit_ratio'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0,
    }
    
    return stats, best_trade, worst_trade


def print_statistics(stats, best_trade, worst_trade):
    """Print formatted statistics"""
    
    print("\n" + "="*60)
    print("📊 TRADING STATISTICS")
    print("="*60)
    
    print(f"\n📈 Overall Performance")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Winning Trades: {stats['winning_trades']}")
    print(f"  Losing Trades: {stats['losing_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.2f}%")
    print(f"  Total Profit: {stats['total_profit_pct']:.2f}%")
    print(f"  Average Profit: {stats['avg_profit_pct']:.2f}%")
    print(f"  Average Win: {stats['avg_win_pct']:.2f}%")
    print(f"  Average Loss: {stats['avg_loss_pct']:.2f}%")
    print(f"  Average Duration: {stats['avg_duration']}")
    
    print(f"\n📊 Long vs Short")
    print(f"  Long Trades: {stats['long_trades']} (Win Rate: {stats['long_win_rate']:.2f}%)")
    print(f"  Short Trades: {stats['short_trades']} (Win Rate: {stats['short_win_rate']:.2f}%)")
    
    if best_trade is not None:
        print(f"\n🏆 Best Trade")
        print(f"  Pair: {best_trade['pair']}")
        print(f"  Entry: {best_trade['entry_time']}")
        print(f"  Profit: {best_trade['profit_ratio']*100:.2f}%")
        print(f"  Side: {best_trade['side'].upper()}")
    
    if worst_trade is not None:
        print(f"\n💔 Worst Trade")
        print(f"  Pair: {worst_trade['pair']}")
        print(f"  Entry: {worst_trade['entry_time']}")
        print(f"  Loss: {worst_trade['profit_ratio']*100:.2f}%")
        print(f"  Side: {worst_trade['side'].upper()}")
    
    print("\n" + "="*60 + "\n")


def analyze_indicators(df):
    """Analyze indicator effectiveness"""
    
    print("\n" + "="*60)
    print("🔍 INDICATOR ANALYSIS")
    print("="*60)
    
    # RSI analysis
    winning_rsi_5m_entry = df[df['profit_ratio'] > 0]['entry_rsi_5m'].mean()
    losing_rsi_5m_entry = df[df['profit_ratio'] <= 0]['entry_rsi_5m'].mean()
    
    print(f"\n📊 RSI 5m at Entry")
    print(f"  Winning trades avg: {winning_rsi_5m_entry:.2f}")
    print(f"  Losing trades avg: {losing_rsi_5m_entry:.2f}")
    
    # ADX analysis
    winning_adx_entry = df[df['profit_ratio'] > 0]['entry_adx_5m'].mean()
    losing_adx_entry = df[df['profit_ratio'] <= 0]['entry_adx_5m'].mean()
    
    print(f"\n📊 ADX 5m at Entry")
    print(f"  Winning trades avg: {winning_adx_entry:.2f}")
    print(f"  Losing trades avg: {losing_adx_entry:.2f}")
    
    # MA Signal analysis
    winning_buy_signals = df[df['profit_ratio'] > 0]['entry_buy_signal_count'].mean()
    losing_buy_signals = df[df['profit_ratio'] <= 0]['entry_buy_signal_count'].mean()
    
    print(f"\n📊 Buy Signal Count at Entry")
    print(f"  Winning trades avg: {winning_buy_signals:.2f}")
    print(f"  Losing trades avg: {losing_buy_signals:.2f}")
    
    # SuperTrend alignment
    df['st_aligned_entry'] = (
        (df['entry_st_7_3_direction'] == 1).astype(int) +
        (df['entry_st_10_2_direction'] == 1).astype(int) +
        (df['entry_st_21_7_direction'] == 1).astype(int)
    )
    
    winning_st_aligned = df[df['profit_ratio'] > 0]['st_aligned_entry'].mean()
    losing_st_aligned = df[df['profit_ratio'] <= 0]['st_aligned_entry'].mean()
    
    print(f"\n📊 SuperTrend Alignment at Entry")
    print(f"  Winning trades avg: {winning_st_aligned:.2f}")
    print(f"  Losing trades avg: {losing_st_aligned:.2f}")
    
    print("\n" + "="*60 + "\n")


def plot_results(df, save_fig=True):
    """Generate visualizations"""
    
    if len(df) == 0:
        print("No trades to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Trading Performance Analysis', fontsize=16)
    
    # 1. Profit distribution
    axes[0, 0].hist(df['profit_ratio'] * 100, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Profit %')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Profit Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cumulative profit
    df_sorted = df.sort_values('exit_time')
    cumulative_profit = (df_sorted['profit_ratio'] * 100).cumsum()
    axes[0, 1].plot(cumulative_profit.values, linewidth=2)
    axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Trade Number')
    axes[0, 1].set_ylabel('Cumulative Profit %')
    axes[0, 1].set_title('Cumulative Profit Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Win rate by pair
    if 'pair' in df.columns:
        pair_stats = df.groupby('pair').agg({
            'profit_ratio': lambda x: (x > 0).sum() / len(x) * 100
        }).sort_values('profit_ratio', ascending=False)
        
        pair_stats.plot(kind='bar', ax=axes[1, 0], legend=False)
        axes[1, 0].set_xlabel('Pair')
        axes[1, 0].set_ylabel('Win Rate %')
        axes[1, 0].set_title('Win Rate by Pair')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. RSI vs Profit scatter
    axes[1, 1].scatter(df['entry_rsi_5m'], df['profit_ratio'] * 100, alpha=0.5)
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Entry RSI 5m')
    axes[1, 1].set_ylabel('Profit %')
    axes[1, 1].set_title('RSI vs Profit')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        output_dir = Path('user_data/logs')
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"📊 Chart saved to: {filepath}")
    
    plt.show()


def main():
    """Main analysis function"""
    
    print("\n🔍 Loading trade data...")
    df = load_trades()
    
    if df is None or len(df) == 0:
        print("❌ No trades found to analyze")
        return
    
    print(f"✅ Loaded {len(df)} trades\n")
    
    # Calculate and print statistics
    stats, best_trade, worst_trade = calculate_statistics(df)
    print_statistics(stats, best_trade, worst_trade)
    
    # Analyze indicators
    analyze_indicators(df)
    
    # Generate plots
    print("📊 Generating visualizations...")
    plot_results(df)
    
    # Export summary to Excel (optional)
    try:
        summary_path = Path('user_data/logs/trade_summary.xlsx')
        
        with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All Trades', index=False)
            
            # Winning trades
            df[df['profit_ratio'] > 0].to_excel(writer, sheet_name='Winning Trades', index=False)
            
            # Losing trades
            df[df['profit_ratio'] <= 0].to_excel(writer, sheet_name='Losing Trades', index=False)
        
        print(f"📊 Summary exported to: {summary_path}")
    except Exception as e:
        print(f"⚠️  Could not export to Excel: {e}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
