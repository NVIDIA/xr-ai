# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Performance Visualization Script

This script reads the performance_analysis.csv and creates comprehensive visualizations
showing the breakdown of timing components and performance improvements across different
configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_process_data(csv_path):
    """Load and process the performance data."""
    df = pd.read_csv(csv_path)
    
    # Extract configuration names and components
    configs = []
    components = ['Caption', 'Detection', 'Segmentation', 'CLIP', 'Mapping']
    
    for col in df.columns:
        if col != 'scene':
            config_name = col.split('_')[0:-1]  # Remove the component part
            config_name = '_'.join(config_name)
            if config_name not in configs:
                configs.append(config_name)
    
    return df, configs, components

# def create_stacked_bar_chart(df, configs, components, output_dir):
#     """Create stacked bar charts showing time breakdown for each configuration."""
#     scenes = df['scene'].tolist()
    
#     fig, axes = plt.subplots(1, 3, figsize=(20, 8))
#     fig.suptitle('Performance Breakdown by Configuration\n(Stacked Time Components)', fontsize=16, fontweight='bold')
    
#     colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
#     for i, config in enumerate(configs):
#         ax = axes[i]
        
#         # Prepare data for this configuration
#         data_matrix = []
#         for component in components:
#             col_name = f"{config}_{component}"
#             if col_name in df.columns:
#                 data_matrix.append(df[col_name].tolist())
#             else:
#                 data_matrix.append([0] * len(scenes))
        
#         data_matrix = np.array(data_matrix)
        
#         # Create stacked bar chart
#         bottom = np.zeros(len(scenes))
#         for j, component in enumerate(components):
#             ax.bar(scenes, data_matrix[j], bottom=bottom, label=component, 
#                   color=colors[j], alpha=0.8)
#             bottom += data_matrix[j]
        
#         ax.set_title(f'{config}', fontsize=14, fontweight='bold')
#         ax.set_ylabel('Time (ms)', fontsize=12)
#         ax.tick_params(axis='x', rotation=45)
#         ax.grid(True, alpha=0.3)
        
#         # Add total time annotations
#         totals = bottom
#         for k, (scene, total) in enumerate(zip(scenes, totals)):
#             ax.annotate(f'{total:.0f}ms', (k, total), 
#                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
#     # Add legend to the last subplot
#     axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig(output_dir / 'stacked_performance_breakdown.png', dpi=300, bbox_inches='tight')
#     plt.show()

# def create_grouped_comparison(df, configs, components, output_dir):
#     """Create grouped bar charts comparing each component across configurations."""
#     scenes = df['scene'].tolist()
    
#     fig, axes = plt.subplots(2, 3, figsize=(20, 12))
#     fig.suptitle('Component-wise Performance Comparison Across Configurations', fontsize=16, fontweight='bold')
#     axes = axes.flatten()
    
#     x = np.arange(len(scenes))
#     width = 0.25
#     colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
#     for comp_idx, component in enumerate(components):
#         ax = axes[comp_idx]
        
#         for i, config in enumerate(configs):
#             col_name = f"{config}_{component}"
#             if col_name in df.columns:
#                 values = df[col_name].tolist()
#                 ax.bar(x + i * width, values, width, label=config, 
#                       color=colors[i], alpha=0.8)
        
#         ax.set_title(f'{component} Performance', fontsize=14, fontweight='bold')
#         ax.set_ylabel('Time (ms)', fontsize=12)
#         ax.set_xlabel('Scene', fontsize=12)
#         ax.set_xticks(x + width)
#         ax.set_xticklabels(scenes, rotation=45)
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     # Remove the empty subplot
#     fig.delaxes(axes[5])
    
#     plt.tight_layout()
#     plt.savefig(output_dir / 'grouped_component_comparison.png', dpi=300, bbox_inches='tight')
#     plt.show()

# def create_performance_improvement_chart(df, configs, components, output_dir):
#     """Create charts showing performance improvements relative to baseline."""
#     baseline_config = configs[0]  # Assume first config is baseline
#     scenes = df['scene'].tolist()
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
#     fig.suptitle('Performance Improvements Relative to Baseline', fontsize=16, fontweight='bold')
    
#     # Calculate total times for each configuration
#     total_times = {}
#     for config in configs:
#         total_time = np.zeros(len(scenes))
#         for component in components:
#             col_name = f"{config}_{component}"
#             if col_name in df.columns:
#                 total_time += df[col_name].values
#         total_times[config] = total_time
    
#     # Plot 1: Speedup factors
#     speedups = {}
#     for i, config in enumerate(configs[1:], 1):  # Skip baseline
#         speedup = total_times[baseline_config] / total_times[config]
#         speedups[config] = speedup
#         ax1.plot(scenes, speedup, marker='o', linewidth=2, markersize=8, label=config)
    
#     ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
#     ax1.set_title('Speedup Factors vs Baseline', fontsize=14, fontweight='bold')
#     ax1.set_ylabel('Speedup Factor', fontsize=12)
#     ax1.set_xlabel('Scene', fontsize=12)
#     ax1.tick_params(axis='x', rotation=45)
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
#     ax1.set_ylim(bottom=0.5)
    
#     # Plot 2: Time savings in milliseconds
#     for i, config in enumerate(configs[1:], 1):
#         time_saved = total_times[baseline_config] - total_times[config]
#         ax2.bar([x + i*0.3 for x in range(len(scenes))], time_saved, 
#                width=0.3, label=config, alpha=0.8)
    
#     ax2.set_title('Time Savings vs Baseline', fontsize=14, fontweight='bold')
#     ax2.set_ylabel('Time Saved (ms)', fontsize=12)
#     ax2.set_xlabel('Scene', fontsize=12)
#     ax2.set_xticks([x + 0.3 for x in range(len(scenes))])
#     ax2.set_xticklabels(scenes, rotation=45)
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(output_dir / 'performance_improvements.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     return speedups

def create_scene_clustered_stacked_bars(df, configs, components, output_dir):
    """Create clustered stacked bar charts - one cluster per scene, each cluster has 3 stacked bars."""
    scenes = df['scene'].tolist()
    
    fig, ax = plt.subplots(1, 1, figsize=(28, 13.5))
    
    # More distinct colors: Red, Orange, Purple, Green, Dark Maroon
    colors = ['#E74C3C', '#F39C12', '#9B59B6', '#27AE60', '#922B21']
    # More distinct colors for each configuration
    config_colors = ['#34495E', '#E74C3C', '#3498DB']
    
    n_scenes = len(scenes)
    n_configs = len(configs)
    bar_width = 0.25
    
    # Create positions for bars with better spacing
    scene_positions = np.arange(n_scenes) * 1.0  # Proper space between scene groups
    
    # For each configuration, create stacked bars
    for config_idx, config in enumerate(configs):
        # Calculate positions for this configuration's bars
        positions = scene_positions + (config_idx - 1) * bar_width
        
        # Prepare data for this configuration
        data_matrix = []
        for component in components:
            col_name = f"{config}_{component}"
            if col_name in df.columns:
                data_matrix.append(df[col_name].tolist())
            else:
                data_matrix.append([0] * len(scenes))
        
        data_matrix = np.array(data_matrix)
        
        # Create stacked bars for this configuration
        bottom = np.zeros(len(scenes))
        for comp_idx, component in enumerate(components):
            bars = ax.bar(positions, data_matrix[comp_idx], bar_width, 
                         bottom=bottom, 
                         label=f'{component}' if config_idx == 0 else '',
                         color=colors[comp_idx], alpha=0.8, 
                         edgecolor='white', linewidth=1)
            bottom += data_matrix[comp_idx]
        
        # Add total time annotations on top of each stacked bar
        totals = bottom
        # for i, (pos, total) in enumerate(zip(positions, totals)):
        #     ax.annotate(f'{total:.0f}ms', (pos, total), 
        #                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Customize the plot
    # ax.set_title('Performance Breakdown by Scene and Configuration', 
    #             fontsize=18, fontweight='bold', pad=30)
    # ax.set_xlabel('Scene', fontsize=24, fontweight='bold')  # Removed as it's evident
    ax.set_ylabel('Processing Time (ms)', fontsize=26, fontweight='bold')
    
    # Set x-axis with scene names
    ax.set_xticks(scene_positions)
    ax.set_xticklabels(scenes, fontsize=22, fontweight='bold')
    
    # Add simple text labels below each bar
    for scene_idx, scene in enumerate(scenes):
        for config_idx, config in enumerate(configs):
            x_pos = scene_positions[scene_idx] + (config_idx - 1) * bar_width
            # Single letter labels positioned below the bars
            short_labels = ['B', 'P', 'SD']
            ax.text(x_pos, -35, short_labels[config_idx], 
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='circle,pad=0.2', 
                            facecolor=config_colors[config_idx], 
                            edgecolor='white',
                            linewidth=1, alpha=0.9))
    
    # Create legends optimized for slide format
    # Components legend - move to right side in vertical column
    # legend1 = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
    #                    fontsize=24, ncol=1, 
    #                    frameon=False,
    #                 #    title='Components', title_fontsize=24
    #                    )
    # for text in legend1.get_texts():
    #     text.set_fontweight('bold')
    # legend1.get_title().set_fontweight('bold')
    
    # Configurations legend - keep at bottom but make more compact
    config_handles = []
    config_labels = ['B: Baseline', 'P: Object-centric Parallelization', 'SD: Object-centric SelectiveDownsampling']
    for config_idx, config in enumerate(config_labels):
        handle = plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=config_colors[config_idx],
                           markeredgecolor='white', markeredgewidth=1,
                           markersize=20, alpha=0.9, linestyle='None')
        config_handles.append(handle)
    
    # legend2 = ax.legend(config_handles, config_labels, 
    #                    bbox_to_anchor=(0.5, -0.03), loc='upper center',
    #                    fontsize=24, ncol=3, columnspacing=1.5,
    #                    frameon=False,
    #                 #    title='Configurations'
    #                    )
    # for text in legend2.get_texts():
    #     text.set_fontweight('bold')
    # legend2.get_title().set_fontweight('bold')
    
    # Add both legends
    # ax.add_artist(legend1)
    
    # Set white background and remove grid
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)
    
    # Make all tick labels bold and increase y-tick size
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(30)
    
    # Add vertical separators between scene groups
    for i in range(1, len(scenes)):
        separator_x = (scene_positions[i] + scene_positions[i-1]) / 2
        ax.axvline(x=separator_x, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Adjust margins and y-limits for better appearance and space for labels
    ax.margins(x=0.02)
    # ax.set_ylim(bottom=-50)  # Make room for labels below the bars
    
    # Adjust layout to accommodate right-side legend and bottom configurations legend
    plt.subplots_adjust(bottom=0.08, left=0.06, right=0.85, top=0.98)
    plt.savefig(output_dir / 'scene_clustered_stacked_bars.png', dpi=300, bbox_inches='tight')
    plt.show()  # Display the plot
    plt.close()  # Close the figure to free memory

# def create_heatmap_comparison(df, configs, components, output_dir):
#     """Create a heatmap showing performance across scenes and configurations."""
#     scenes = df['scene'].tolist()
    
#     # Calculate total times for heatmap
#     heatmap_data = []
#     for config in configs:
#         config_totals = []
#         for _, row in df.iterrows():
#             total = 0
#             for component in components:
#                 col_name = f"{config}_{component}"
#                 if col_name in df.columns:
#                     total += row[col_name]
#             config_totals.append(total)
#         heatmap_data.append(config_totals)
    
#     heatmap_df = pd.DataFrame(heatmap_data, columns=scenes, index=configs)
    
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(heatmap_df, annot=True, fmt='.0f', cmap='RdYlBu_r', 
#                 cbar_kws={'label': 'Total Time (ms)'})
#     plt.title('Performance Heatmap: Total Processing Time by Configuration and Scene', 
#               fontsize=14, fontweight='bold')
#     plt.ylabel('Configuration', fontsize=12)
#     plt.xlabel('Scene', fontsize=12)
#     plt.tight_layout()
#     plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
#     plt.show()

def print_performance_summary(df, configs, components, speedups):
    """Print a summary of performance improvements."""
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    
    # Calculate average total times
    avg_totals = {}
    for config in configs:
        total_times = []
        for _, row in df.iterrows():
            total = sum(row[f"{config}_{comp}"] for comp in components 
                       if f"{config}_{comp}" in df.columns)
            total_times.append(total)
        avg_totals[config] = np.mean(total_times)
    
    baseline_avg = avg_totals[configs[0]]
    print(f"\nAverage Total Processing Times:")
    for config, avg_time in avg_totals.items():
        improvement = ((baseline_avg - avg_time) / baseline_avg) * 100 if config != configs[0] else 0
        print(f"  {config}: {avg_time:.1f}ms ({improvement:+.1f}%)")
    
    print(f"\nBest Performance Improvements:")
    for config, speedup_values in speedups.items():
        max_speedup = np.max(speedup_values)
        best_scene_idx = np.argmax(speedup_values)
        best_scene = df['scene'].iloc[best_scene_idx]
        print(f"  {config}: {max_speedup:.2f}x speedup on {best_scene}")
    
    print(f"\nComponent Analysis (Average across all scenes):")
    for component in components:
        print(f"\n  {component}:")
        for config in configs:
            col_name = f"{config}_{component}"
            if col_name in df.columns:
                avg_time = df[col_name].mean()
                print(f"    {config}: {avg_time:.1f}ms")

def main():
    """Main function to run all visualizations."""
    # File paths
    csv_path = Path("performance_analysis.csv")
    output_dir = Path("performance_plots")
    output_dir.mkdir(exist_ok=True)
    
    print("Loading performance data...")
    df, configs, components = load_and_process_data(csv_path)
    
    print(f"Found {len(configs)} configurations: {configs}")
    print(f"Found {len(components)} components: {components}")
    print(f"Processing {len(df)} scenes: {df['scene'].tolist()}")
    
    print("\nGenerating clustered stacked bar chart...")
    
    # Create the main visualization - scene-clustered stacked bars
    create_scene_clustered_stacked_bars(df, configs, components, output_dir)
    
    # Print basic summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # Calculate average total times
    avg_totals = {}
    for config in configs:
        total_times = []
        for _, row in df.iterrows():
            total = sum(row[f"{config}_{comp}"] for comp in components 
                       if f"{config}_{comp}" in df.columns)
            total_times.append(total)
        avg_totals[config] = np.mean(total_times)
    
    baseline_avg = avg_totals[configs[0]]
    print(f"\nAverage Total Processing Times:")
    for config, avg_time in avg_totals.items():
        improvement = ((baseline_avg - avg_time) / baseline_avg) * 100 if config != configs[0] else 0
        print(f"  {config}: {avg_time:.1f}ms ({improvement:+.1f}%)")
    
    print(f"\nAll plots saved to: {output_dir.absolute()}")
    print("Visualization complete!")

if __name__ == "__main__":
    main()
