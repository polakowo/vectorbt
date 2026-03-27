#!/usr/bin/env python3
"""
VectorBT Simple Example - Dual Moving Average Crossover (DMAC)
This is a Python version of the BitcoinDMAC.ipynb example
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import shutil
from datetime import datetime
from PIL import Image

print("=" * 60)
print("VectorBT Example: Dual Moving Average Crossover")
print("=" * 60)

# Load real Bitcoin price data from Yahoo Finance
price = vbt.YFData.download('BTC-USD').get('Close')

print(f"\n📊 Data: {len(price)} hourly price points")
print(f"   Start: ${price.iloc[0]:,.2f}")
print(f"   End: ${price.iloc[-1]:,.2f}")

# Calculate fast and slow moving averages
fast_window = 50
slow_window = 200

fast_ma = vbt.MA.run(price, window=fast_window)
slow_ma = vbt.MA.run(price, window=slow_window)

print(f"\n📈 Moving Averages:")
print(f"   Fast MA ({fast_window}): ${fast_ma.ma.iloc[-1]:,.2f}")
print(f"   Slow MA ({slow_window}): ${slow_ma.ma.iloc[-1]:,.2f}")

# Generate signals: Fast MA crosses above Slow MA = Buy
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

print(f"\n📍 Signals:")
print(f"   Entry signals: {entries.sum()}")
print(f"   Exit signals: {exits.sum()}")

# Run portfolio simulation
portfolio = vbt.Portfolio.from_signals(
    price,
    entries,
    exits,
    init_cash=10000,
    fees=0.001,  # 0.1% fees
    freq='1h'
)

# Display results
print(f"\n💰 Portfolio Performance:")
print(f"   Initial Cash: $10,000.00")
print(f"   Final Value: ${portfolio.final_value():,.2f}")
print(f"   Total Return: {portfolio.total_return():.2%}")
print(f"   Total Trades: {portfolio.trades.count()}")

# Calculate some metrics
returns = portfolio.returns()
print(f"\n📊 Risk Metrics:")
print(f"   Sharpe Ratio: {portfolio.sharpe_ratio():.2f}")
print(f"   Max Drawdown: {portfolio.max_drawdown():.2%}")
print(f"   Win Rate: {portfolio.trades.win_rate():.2%}")

# Create visualization GIF showing price and balance over time
print(f"\n🎬 Generating visualization GIF...")
portfolio_value = portfolio.final_value()
balance_history = portfolio.value()

# Sample every N data points to make GIF manageable (adjust for speed/detail)
sample_rate = max(1, len(price) // 200)  # Target ~200 frames
indices = np.arange(0, len(price), sample_rate)

frames = []
temp_dir = '/tmp/dmac_frames'
os.makedirs(temp_dir, exist_ok=True)

# Set consistent figure size for all frames
fig_width, fig_height = 12, 8
dpi = 80

for idx, i in enumerate(indices):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=dpi)
    
    # Top plot: Price and Moving Averages
    time_range = np.arange(i)
    ax1.plot(time_range, price.values[:i], label='BTC Price', color='black', linewidth=2)
    if i > fast_window:
        ax1.plot(time_range, fast_ma.ma.values[:i], label=f'Fast MA ({fast_window})', 
                color='blue', alpha=0.7, linewidth=1.5)
    if i > slow_window:
        ax1.plot(time_range, slow_ma.ma.values[:i], label=f'Slow MA ({slow_window})', 
                color='red', alpha=0.7, linewidth=1.5)
    
    # Mark entry/exit signals
    entry_mask = entries.values[:i] if hasattr(entries, 'values') else entries[:i]
    exit_mask = exits.values[:i] if hasattr(exits, 'values') else exits[:i]
    
    if entry_mask.any():
        entry_idx = np.where(entry_mask)[0]
        ax1.scatter(entry_idx, price.values[entry_idx], 
                   color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    if exit_mask.any():
        exit_idx = np.where(exit_mask)[0]
        ax1.scatter(exit_idx, price.values[exit_idx], 
                   color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_ylabel('Price (USD)')
    ax1.set_title('Bitcoin Price with Moving Averages and Trading Signals')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Portfolio Balance
    balance_values = balance_history.values[:i] if hasattr(balance_history, 'values') else balance_history[:i]
    ax2.plot(time_range, balance_values, label='Portfolio Balance', 
            color='green', linewidth=2)
    ax2.fill_between(time_range, balance_values, 10000, alpha=0.3, color='green')
    ax2.axhline(y=10000, color='gray', linestyle='--', label='Initial Cash', alpha=0.5)
    ax2.set_ylabel('Balance (USD)')
    ax2.set_xlabel('Time Index')
    ax2.set_title('Portfolio Balance Over Time')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add progress indicator
    progress_pct = (i / len(price)) * 100
    fig.suptitle(f'DMAC Strategy Backtest - Progress: {progress_pct:.1f}%', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save frame
    frame_path = os.path.join(temp_dir, f'frame_{idx:04d}.png')
    plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
    frames.append(imageio.imread(frame_path))
    plt.close(fig)

# Create GIF from frames
gif_path = 'dmac_backtest.gif'
print(f"   Creating GIF with {len(frames)} frames...")

# Ensure all frames have the same size by resizing to the most common dimensions
frame_sizes = [f.shape for f in frames]
# Use first frame size as target (usually most consistent)
target_size = frame_sizes[0][:2]  # Get height, width

# Resize all frames to target size
frames_resized = []
for f in frames:
    if f.shape[:2] != target_size:
        img = Image.fromarray(f)
        img_resized = img.resize((target_size[1], target_size[0]))  # PIL uses width, height
        frames_resized.append(np.array(img_resized))
    else:
        frames_resized.append(f)

imageio.mimsave(gif_path, frames_resized, duration=0.1)  # 100ms per frame

# Clean up temp files
shutil.rmtree(temp_dir)

print(f"✅ GIF saved to: {gif_path}")

print("\n" + "=" * 60)
print("✅ Example completed successfully!")
print("=" * 60)
print("\nTo run this:")
print("  cd ~/projects/quant")
print("  source vbt-env/bin/activate")
print("  python3 examples/dmac_example.py")
