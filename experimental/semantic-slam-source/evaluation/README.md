<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Evaluation Module

This directory contains evaluation tools for assessing the performance of the Semantic SLAM system, particularly for semantic segmentation and object mapping quality on the Replica dataset.

## Overview

The evaluation module provides comprehensive tools for:
- **Quantitative evaluation** of semantic segmentation accuracy using standard metrics (mIoU, precision, recall, F1-score)
- **3D point cloud comparison** between predicted and ground truth semantic maps
- **Interactive visualization** of mapping results with multiple viewing modes
- **Performance analysis** across different configurations and scenes

## Files

### Core Evaluation Scripts

#### `eval_replica_semseg.py`
**Main evaluation script for Replica dataset semantic segmentation**

Computes semantic segmentation metrics by comparing predicted object maps against ground truth semantic annotations.

**Usage:**
```bash
python evaluation/eval_replica_semseg.py \
    --replica_root ~/rdata/Replica/ \
    --replica_semantic_root ~/rdata/Replica-semantic/ \
    --pred_exp_name "your_experiment_name" \
    --n_exclude 6 \
    --clip_model_name mobileclip
```

**Key Features:**
- Loads predicted object maps from SLAM results (`.pkl.gz` files)
- Compares against ground truth semantic point clouds
- Supports different evaluation modes (exclude 1/4/6 classes)
- Uses CLIP embeddings for object classification
- Outputs detailed metrics per scene and overall performance

**Arguments:**
- `--replica_root`: Path to Replica dataset root directory
- `--replica_semantic_root`: Path to Replica semantic annotations
- `--pred_exp_name`: Name of your experiment (used to find result files)
- `--n_exclude`: Number of classes to exclude from evaluation:
  - `1`: exclude "other" 
  - `4`: exclude "other", "floor", "wall", "ceiling"
  - `6`: exclude "other", "floor", "wall", "ceiling", "door", "window"
- `--clip_model_name`: CLIP model variant (`mobileclip` or `origclip`)

**Output:**
- `results/{exp_name}/replica_ex{n}_results.csv`: Per-scene metrics
- `results/{exp_name}/replica_ex{n}_conf_matrices.pkl`: Confusion matrices

#### `eval.py`
**Core evaluation utilities and metrics computation**

Contains fundamental evaluation functions used by other scripts:

**Key Functions:**
- `compute_pred_gt_associations()`: Finds nearest neighbor associations between predicted and ground truth point clouds
- `compute_confmatrix()`: Builds confusion matrix for semantic classification
- `compute_metrics()`: Calculates standard metrics (IoU, precision, recall, F1-score) from confusion matrix

**Metrics Computed:**
- **mIoU**: Mean Intersection over Union across all classes
- **fmIoU**: Frequency-weighted mean IoU
- **Precision/Recall/F1**: Per-class and mean values
- **Accuracy thresholds**: Number of classes above IoU thresholds (0.15, 0.25, 0.50, 0.75)

#### `visualize_results.py`
**Interactive 3D visualization tool for mapping results**

Provides rich interactive visualization of SLAM mapping results with multiple viewing modes and real-time querying capabilities.

**Usage:**
```bash
# Visualize mapping results
python evaluation/visualize_results.py \
    --result_path path/to/results.pkl.gz \
    --rgb_pcd_path path/to/rgb_cloud.h5 \
    --edge_file path/to/scene_graph.json

# Quick visualization without CLIP (faster startup)
python evaluation/visualize_results.py \
    --result_path path/to/results.pkl.gz \
    --no_clip
```

**Interactive Controls:**
- **B**: Toggle background objects visibility
- **S**: Toggle global RGB point cloud
- **C**: Color objects by semantic class
- **R**: Color objects by original RGB
- **I**: Color objects by instance ID
- **F**: Color by CLIP similarity to text query (requires CLIP model)
- **G**: Toggle scene graph visualization (requires edge file)
- **V**: Save current camera view parameters

**Features:**
- Real-time CLIP-based querying: Type natural language queries to highlight relevant objects
- Multiple coloring modes: semantic classes, RGB, instance IDs, similarity scores
- Scene graph visualization: Shows spatial relationships between objects
- Background object filtering: Toggle detection-based vs. segmentation-based objects
- Camera view saving: Export view parameters for consistent visualization

## Evaluation Workflow

### 1. Generate Mapping Results
First, run the SLAM system to generate mapping results:
```bash
# Run SLAM on Replica scenes
python server/main.py --dataset_type replica --useDataset room0 --config config/your_config.yaml
```

This produces result files in: `~/rdata/Replica/room0/pcd_saves/full_pcd_{exp_name}*.pkl.gz`

### 2. Quantitative Evaluation
Evaluate semantic segmentation performance:
```bash
python evaluation/eval_replica_semseg.py \
    --replica_root ~/rdata/Replica/ \
    --replica_semantic_root ~/rdata/Replica-semantic/ \
    --pred_exp_name "your_experiment_name" \
    --n_exclude 6
```

### 3. Qualitative Analysis
Visualize and inspect results interactively:
```bash
python evaluation/visualize_results.py \
    --result_path ~/rdata/Replica/room0/pcd_saves/full_pcd_your_experiment.pkl.gz \
    --rgb_pcd_path ~/rdata/Replica/room0/rgb_cloud.h5
```

### 4. Performance Analysis
Compare results across different configurations:
```bash
# Evaluate multiple experiments
for exp in baseline parallelization_mobileclip; do
    python evaluation/eval_replica_semseg.py --pred_exp_name $exp --n_exclude 6
done

# Analyze results
python -c "
import pandas as pd
import glob
results = []
for f in glob.glob('results/*/replica_ex6_results.csv'):
    df = pd.read_csv(f)
    df['experiment'] = f.split('/')[1]
    results.append(df)
combined = pd.concat(results)
print(combined.groupby('experiment')[['miou', 'mrecall', 'mprecision']].mean())
"
```

## Understanding the Results

### Metrics Interpretation

**mIoU (mean Intersection over Union)**: 
- Primary metric for semantic segmentation quality
- Range: 0-100%, higher is better
- Good performance: >50%, excellent: >70%

**Precision/Recall/F1-score**:
- Precision: How many predicted pixels of a class are correct
- Recall: How many ground truth pixels of a class are detected  
- F1-score: Harmonic mean of precision and recall

**fmIoU (frequency-weighted mIoU)**:
- Weights classes by their frequency in the dataset
- Better reflects performance on common vs. rare classes

### Class Exclusion Strategies

Different `n_exclude` values test robustness:

- **n_exclude=1**: Full evaluation (exclude only "other" class)
- **n_exclude=4**: Focus on objects (exclude structural elements)
- **n_exclude=6**: Object-only evaluation (exclude all background)

Higher exclusion typically yields better scores but tests fewer capabilities.

### Typical Performance Ranges

Based on Replica dataset evaluation:

| Metric | Good | Excellent |
|--------|------|-----------|
| mIoU (n_exclude=6) | 45-60% | >65% |
| mIoU (n_exclude=4) | 35-50% | >55% |
| mIoU (n_exclude=1) | 25-40% | >45% |

## Configuration and Customization

### Custom Evaluation Metrics
Extend `eval.py` to add custom metrics:

```python
def compute_custom_metrics(confmatrix, class_names):
    # Add your custom metric computation
    custom_score = your_metric_function(confmatrix)
    return {"custom_metric": custom_score}
```

### Visualization Customization
Modify `visualize_results.py` for custom views:

```python
def custom_coloring_mode(vis):
    # Implement your custom coloring scheme
    for i, pcd in enumerate(pcds):
        custom_color = your_color_function(objects[i])
        pcd.colors = o3d.utility.Vector3dVector(custom_color)
    
# Register custom key callback
vis.register_key_callback(ord("X"), custom_coloring_mode)
```

### Batch Evaluation
For systematic evaluation across multiple scenes/configs:

```bash
#!/bin/bash
# evaluate_all.sh
CONFIGS=("baseline" "parallelization_mobileclip" "your_config")
SCENES=("room0" "room1" "room2" "office0" "office1")

for config in "${CONFIGS[@]}"; do
    for scene in "${SCENES[@]}"; do
        echo "Evaluating $config on $scene"
        python evaluation/eval_replica_semseg.py \
            --pred_exp_name "${config}" \
            --n_exclude 6 \
            --replica_root ~/rdata/Replica/
    done
done
```

## Dependencies

The evaluation module requires:
- **Core**: `numpy`, `torch`, `pandas`
- **3D Processing**: `open3d`, `gradslam` 
- **AI Models**: `open_clip_torch`
- **Metrics**: `chamferdist` (for point cloud distances)
- **Visualization**: `matplotlib`, `distinctipy`

All dependencies are installed with the main package installation (`pip install -e .`).

## Troubleshooting

### Common Issues

**"No result found" error:**
- Check that `pred_exp_name` matches your experiment name
- Verify result files exist in `{replica_root}/{scene}/pcd_saves/`
- Try different path patterns (with/without `.pkl.gz` extension)

**CLIP model initialization slow:**
- Use `--no_clip` flag for faster debugging
- Consider using `mobileclip` instead of `origclip` for speed

**Memory issues with large point clouds:**
- Increase voxel downsampling in `visualize_results.py`
- Process scenes individually rather than in batch

**Visualization performance:**
- Reduce point cloud density with `pcd.voxel_down_sample(voxel_size)`
- Disable background objects if not needed
- Use smaller CLIP batch sizes

### Performance Tips

1. **Faster evaluation**: Use `mobileclip` instead of `origclip`
2. **Memory efficiency**: Process one scene at a time for large datasets
3. **Debugging**: Use `--no_clip` flag to skip model loading
4. **Batch processing**: Use shell scripts for systematic evaluation

## Integration with Main System

The evaluation module integrates seamlessly with the main SLAM system:

1. **Result Format**: Automatically reads standard SLAM output format (`.pkl.gz` files)
2. **Configuration**: Uses same CLIP models and class definitions as main system
3. **Logging**: Results can be integrated with performance logging system
4. **Visualization**: Supports same object representations as real-time system

This ensures evaluation results directly reflect real-world system performance.
