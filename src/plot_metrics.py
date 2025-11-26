"""
Visualization script for performance metrics
Reads metadata jsonl files and generates charts:
1. Time vs Token throughput (with optional smoothing)
2. Time vs Prefix length / Total tokens
3. Time vs Active trajectories
4. Comparison across different batch sizes
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_metadata(jsonl_file):
    """Load metadata from jsonl file"""
    data = []
    with open(jsonl_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def smooth(data, window_size=10):
    """Sliding window smoothing"""
    if len(data) < window_size:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed

def plot_per_batch_size(metadata, output_dir="plots"):
    """Generate separate plots for each batch size"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by batch_size
    batch_data = {}
    for point in metadata:
        bs = point["batch_size"]
        if bs not in batch_data:
            batch_data[bs] = []
        batch_data[bs].append(point)
    
    # Plot for each batch_size
    for bs, points in batch_data.items():
        # Sort by timestamp
        points = sorted(points, key=lambda x: x["timestamp"])
        
        times = [p["timestamp"] for p in points]
        token_throughput = [p["token_throughput"] for p in points]
        total_tokens = [p["total_tokens_generated"] for p in points]
        active_trajs = [p["active_trajs"] for p in points]
        completed_trajs = [p["completed_trajs"] for p in points]
        
        # Smoothing
        token_throughput_smooth = smooth(token_throughput, window_size=20)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Time vs Token throughput
        ax1.plot(times, token_throughput, 'o', alpha=0.2, markersize=2, label='Raw data')
        ax1.plot(times, token_throughput_smooth, 'r-', linewidth=2, label='Smoothed (window=20)')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Token Throughput (tokens/s)', fontsize=12)
        ax1.set_title(f'Batch Size {bs} - Token Throughput Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time vs Total tokens generated
        ax2.plot(times, total_tokens, 'b-', linewidth=2)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Cumulative Tokens Generated', fontsize=12)
        ax2.set_title(f'Batch Size {bs} - Cumulative Token Generation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Time vs Active/Completed trajectories
        ax3.plot(times, active_trajs, 'g-', linewidth=2, label='Active')
        ax3.plot(times, completed_trajs, 'orange', linewidth=2, label='Completed')
        ax3.set_xlabel('Time (seconds)', fontsize=12)
        ax3.set_ylabel('Number of Trajectories', fontsize=12)
        ax3.set_title(f'Batch Size {bs} - Trajectory Status Over Time', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/batch_size_{bs}.png', dpi=150)
        plt.close()
        
        print(f"✓ Generated: {output_dir}/batch_size_{bs}.png")

def plot_comparison(metadata, output_dir="plots"):
    """Generate comparison plots across all batch sizes"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by batch_size
    batch_data = {}
    for point in metadata:
        bs = point["batch_size"]
        if bs not in batch_data:
            batch_data[bs] = []
        batch_data[bs].append(point)
    
    # Create comparison figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_data)))
    
    for (bs, points), color in zip(sorted(batch_data.items()), colors):
        points = sorted(points, key=lambda x: x["timestamp"])
        
        times = [p["timestamp"] for p in points]
        token_throughput = [p["token_throughput"] for p in points]
        total_tokens = [p["total_tokens_generated"] for p in points]
        
        # Smoothing
        token_throughput_smooth = smooth(token_throughput, window_size=20)
        
        # Plot 1: Token throughput comparison
        ax1.plot(times, token_throughput_smooth, '-', linewidth=2, 
                label=f'BS={bs}', color=color)
        
        # Plot 2: Total tokens comparison
        ax2.plot(times, total_tokens, '-', linewidth=2, 
                label=f'BS={bs}', color=color)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Token Throughput (tokens/s)', fontsize=12)
    ax1.set_title('Token Throughput Comparison Across Batch Sizes (Smoothed)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Cumulative Tokens Generated', fontsize=12)
    ax2.set_title('Cumulative Token Generation Comparison Across Batch Sizes', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batch_size_comparison.png', dpi=150)
    plt.close()
    
    print(f"✓ Generated: {output_dir}/batch_size_comparison.png")

def main_plot():
    parser = argparse.ArgumentParser(description='Visualize performance metrics')
    parser.add_argument('metadata_file', help='Path to metadata jsonl file')
    parser.add_argument('--output_dir', default='../plots', help='Output directory for plots')
    parser.add_argument('--separate', action='store_true', help='Generate separate plots per batch size')
    parser.add_argument('--comparison', action='store_true', help='Generate comparison plots')
    args = parser.parse_args()
    
    print("="*70)
    print("Performance Metrics Visualization")
    print("="*70)
    print(f"Input: {args.metadata_file}")
    print(f"Output: {args.output_dir}/")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    metadata = load_metadata(args.metadata_file)
    
    if not metadata:
        print("✗ No valid data found in file!")
        return
    
    print(f"✓ Loaded {len(metadata)} data points")
    
    # Statistics
    batch_sizes = sorted(set(p["batch_size"] for p in metadata))
    print(f"Batch Sizes: {batch_sizes}")
    
    # Calculate summary stats
    for bs in batch_sizes:
        bs_points = [p for p in metadata if p["batch_size"] == bs]
        avg_throughput = np.mean([p["token_throughput"] for p in bs_points])
        print(f"  BS={bs}: {len(bs_points)} points, avg throughput={avg_throughput:.2f} tok/s")
    
    # Generate plots
    if args.separate:
        print("\nGenerating separate plots...")
        plot_per_batch_size(metadata, args.output_dir)
    
    if args.comparison:
        print("\nGenerating comparison plots...")
        plot_comparison(metadata, args.output_dir)
    
    if not args.separate and not args.comparison:
        print("\nGenerating all plots...")
        plot_per_batch_size(metadata, args.output_dir)
        plot_comparison(metadata, args.output_dir)
    
    print("\n✓ Complete!")

if __name__ == '__main__':
    main_plot()
