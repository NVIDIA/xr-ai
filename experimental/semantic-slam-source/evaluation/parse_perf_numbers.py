# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import csv
from pathlib import Path

def parse_profiling_data():
    """
    Parse profiling data from profiling_info_Sept3/ directory and generate CSV output.
    
    The mapping time is calculated as: gobs_creation_time + similarity_time + merging_time
    """
    
    base_dir = Path("/home/rahulsi/Documents/model_trials/semantic-mapping/semantic-slam-server/profiling_info_Sept3")
    
    # Define the three configurations
    configs = {
        "Baseline": "baseline",
        "Object-centric Parallelization": "parallelization_mobileclip", 
        "Object-centric SelectiveDownsampling": "parallelization_mobileclip_objectDownsampling"
    }
    
    # Define scenes
    scenes = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    
    # Collect all data
    all_data = []
    
    for scene in scenes:
        row = [scene]
        
        for config_name, config_dir in configs.items():
            json_file = base_dir / config_dir / scene / f"performance_summary_{scene}.json"
            
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    perf_stats = data["performance_stats"]
                    
                    # Extract timing data (all in milliseconds)
                    caption = perf_stats["caption_time"]["average"]
                    detection = perf_stats["detection_time"]["average"]
                    segmentation = perf_stats["segmentation_time"]["average"]
                    clip = perf_stats["clip_time"]["average"]
                    
                    # Calculate mapping time: gobs + similarity + merging
                    gobs = perf_stats["gobs_creation_time"]["average"]
                    similarity = perf_stats["similarity_time"]["average"]
                    merging = perf_stats["merging_time"]["average"]
                    mapping = gobs + similarity + merging
                    
                    # Add to row
                    row.extend([caption, detection, segmentation, clip, mapping])
            else:
                # Add placeholder if file doesn't exist
                row.extend([0, 0, 0, 0, 0])
        
        all_data.append(row)
    
    # Generate CSV header
    header = ["scene"]
    for config_name in configs.keys():
        header.extend([f"{config_name}_Caption", f"{config_name}_Detection", 
                      f"{config_name}_Segmentation", f"{config_name}_CLIP", f"{config_name}_Mapping"])
    
    # Write to CSV
    output_file = "/home/rahulsi/Documents/model_trials/semantic-mapping/semantic-slam-server/evaluation/performance_analysis.csv"
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(all_data)
    
    print(f"CSV file generated: {output_file}")
    
    # Also print in the commented format style for reference (similar to original format)
    print("\n# Performance Data (in milliseconds):")
    print("# \tBaseline\t\t\t\t\t\tParallelization\t\t\t\t\t\tparallelization_mobileclip_objectDownsampling")
    print("# scene\tCaption\tDetection\tSegmentation\tCLIP\tMapping\tCaption\tDetection\tSegmentation\tCLIP\tMapping\tCaption\tDetection\tSegmentation\tCLIP\tMapping")
    
    for row in all_data:
        # Format the numbers to match the original format (around 6-8 decimal places)
        formatted_row = [row[0]]  # scene name
        for i in range(1, len(row)):
            formatted_row.append(f"{row[i]:.8g}")
        print("# " + "\t".join(formatted_row))

if __name__ == "__main__":
    parse_profiling_data()






