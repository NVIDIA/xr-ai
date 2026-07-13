# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Centralized performance measurement and logging system for SLAM pipeline.

This module provides thread-safe, accurate performance monitoring with:
- High-precision timing measurements
- Centralized logging to files and console
- Graceful shutdown handling
- Statistical summaries
"""

import os
import time
import csv
import json
import threading
import signal
import atexit
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from config.settings import get_config


@dataclass
class FrameTimingData:
    """Structure for per-frame timing measurements."""
    frame_number: int
    timestamp: float
    caption_time: float = 0.0
    detection_time: float = 0.0
    segmentation_time: float = 0.0
    clip_time: float = 0.0
    mapping_time: float = 0.0
    clip_mapping: float = 0.0
    queue_overhead: float = 0.0  # Data transfer and queue waiting time
    total_time: float = 0.0
    server_timestamp: Optional[int] = None
    client_timestamp: Optional[int] = None
    queue_sizes: Optional[Dict[str, int]] = None


class PerformanceManager:
    """
    Thread-safe performance monitoring and logging manager.
    
    Features:
    - High-precision timing with time.perf_counter_ns()
    - Real-time statistics (FPS, averages, percentiles)
    - CSV and JSON logging
    - Configurable console output frequency
    - Graceful shutdown with final summary
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, scene_name: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, scene_name: Optional[str] = None):
        if hasattr(self, 'initialized'):
            return
            
        self.initialized = True
        self.config = get_config()
        
        # Get scene name from parameter, environment variable, or fallback
        if scene_name:
            self.scene_name = scene_name
        else:
            # Check environment variable (set by main process for child processes)
            self.scene_name = os.environ.get('SLAM_SCENE_NAME', None)
        
        print(f"🎯 PerformanceManager scene name: {self.scene_name or 'unknown'}")
        
        # Thread safety
        self._data_lock = threading.Lock()
        self._file_lock = threading.Lock()
        
        # Data storage
        self.frame_data: List[FrameTimingData] = []
        self.recent_frames = deque(maxlen=100)  # For FPS calculation
        self.stats = defaultdict(list)
        
        # Configuration
        self.log_directory = self._setup_log_directory()
        self.console_log_interval = self.config.logging.console_log_interval  # Use config value
        self.frame_count = 0
        self.start_time = time.time()
        self._cleanup_done = False  # Prevent duplicate cleanup calls
        
        # File handles
        self.csv_file = None
        self.json_file = None
        self.setup_logging()
        
        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
        
        print(f"📊 PerformanceManager initialized")
        print(f"   Log directory: {self.log_directory}")
        print(f"   Console updates every {self.console_log_interval} frames")
    
    def _setup_log_directory(self) -> Path:
        """Setup log directory based on configuration."""
        # Use absolute path from config directory, not current working directory
        if hasattr(self.config, 'logging') and hasattr(self.config.logging, 'log_directory'):
            base_dir = Path(self.config.logging.log_directory)
            if not base_dir.is_absolute():
                # Make relative to project root (parent of slam/ directory)
                project_root = Path(__file__).parent.parent.parent
                base_dir = project_root / base_dir
        else:
            # Fallback to project root
            project_root = Path(__file__).parent.parent.parent  
            base_dir = project_root / "logs_performance"
        
        # Get config file name for directory structure: logs_performance/<config_file>/
        config_name = os.environ.get('SLAM_CONFIG_NAME', 'default')
        
        # Create scene-based subdirectory: logs_performance/<config_file>/<scene_name>/
        if self.scene_name:
            # Use specific scene name (e.g., "room0", "office1")
            scene_name = self.scene_name
        else:
            # Access dataset type from config (it's a dataclass, not dict)
            try:
                scene_name = getattr(self.config.dataset, 'type', 'unknown')
            except AttributeError:
                scene_name = 'unknown'
        
        # Three-level directory structure
        log_dir = base_dir / config_name / scene_name
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def setup_logging(self):
        """Initialize log files."""
        # Fixed file names without timestamps (always append to same files)
        self.csv_file = self.log_directory / "frame_timing.csv"
        self.json_file = self.log_directory / f"performance_summary_{self.scene_name}.json"
        
        # Initialize CSV with headers if file doesn't exist
        self._csv_fieldnames = [
            'frame_number', 'timestamp', 'caption_time', 'detection_time',
            'segmentation_time', 'clip_time', 'mapping_time', 'clip_mapping',
            'queue_overhead', 'total_time', 'server_timestamp', 'client_timestamp'
        ]
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
                writer.writeheader()
        
        print(f"📈 Logging to: {self.csv_file}")
        
        # Separate file for client update size contrast (fresh vs full-scene)
        self.update_sizes_file = self.log_directory / "update_sizes.csv"
        self._update_sizes_count = 0
        self._update_sizes_fieldnames = [
            'update_number', 'fresh_update_size_mbits', 'full_scene_update_size_mbits',
            'num_fresh_objects', 'num_total_objects'
        ]
        if not self.update_sizes_file.exists():
            with open(self.update_sizes_file, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self._update_sizes_fieldnames).writeheader()
        print(f"📈 Update sizes logged to: {self.update_sizes_file}")
    
    def log_update_sizes(self, fresh_update_size_mbits: float, full_scene_update_size_mbits: float,
                         num_fresh_objects: int = 0, num_total_objects: int = 0):
        """Log one row to the separate update_sizes.csv file. Thread-safe."""
        with self._file_lock:
            self._update_sizes_count += 1
            try:
                with open(self.update_sizes_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self._update_sizes_fieldnames)
                    writer.writerow({
                        'update_number': self._update_sizes_count,
                        'fresh_update_size_mbits': fresh_update_size_mbits,
                        'full_scene_update_size_mbits': full_scene_update_size_mbits,
                        'num_fresh_objects': num_fresh_objects,
                        'num_total_objects': num_total_objects,
                    })
            except Exception as e:
                print(f"⚠️  Error writing to update_sizes.csv: {e}")
    
    def log_frame_timing(self, frame_number: int, timing_dict: Dict[str, float], 
                        server_timestamp: Optional[int] = None,
                        client_timestamp: Optional[int] = None,
                        queue_sizes: Optional[Dict[str, int]] = None):
        """
        Log timing data for a single frame.
        
        Args:
            frame_number: Frame identifier
            timing_dict: Dictionary with timing measurements in milliseconds
            server_timestamp: Server-side timestamp (nanoseconds)
            client_timestamp: Client-side timestamp (nanoseconds)  
            queue_sizes: Current queue sizes for monitoring
        """
        with self._data_lock:
            current_time = time.time()
            
            # Create timing data structure
            frame_data = FrameTimingData(
                frame_number=frame_number,
                timestamp=current_time,
                caption_time=timing_dict.get('caption_time', 0.0),
                detection_time=timing_dict.get('detection_time', 0.0),
                segmentation_time=timing_dict.get('segmentation_time', 0.0),
                clip_time=timing_dict.get('clip_time', 0.0),
                mapping_time=timing_dict.get('mapping_time', 0.0),
                clip_mapping=timing_dict.get('clip_mapping', 0.0),
                queue_overhead=timing_dict.get('queue_overhead', 0.0),
                total_time=timing_dict.get('total_time', 0.0),
                server_timestamp=server_timestamp,
                client_timestamp=client_timestamp,
                queue_sizes=queue_sizes,
            )
            
            # Store data
            self.frame_data.append(frame_data)
            self.recent_frames.append(current_time)
            self.frame_count += 1
            
            # Update statistics
            for key, value in timing_dict.items():
                if value > 0:  # Only track positive timing values
                    self.stats[key].append(value)
            
            # Write to CSV (thread-safe)
            self._write_csv_row(frame_data)
            
            # Periodic console logging
            if self.frame_count % self.console_log_interval == 0:
                self._log_statistics()
    
    def _write_csv_row(self, frame_data: FrameTimingData):
        """Thread-safe CSV writing."""
        with self._file_lock:
            try:
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
                    row_data = asdict(frame_data)
                    row_data.pop('queue_sizes', None)  # Remove nested dict
                    writer.writerow(row_data)
            except Exception as e:
                print(f"⚠️  Error writing to CSV: {e}")
    
    def _log_statistics(self):
        """Log current performance statistics to console."""
        if not self.recent_frames:
            return
            
        # Calculate FPS over meaningful time span (avoid unrealistic values)
        if len(self.recent_frames) > 10:  # Need at least 10 frames for meaningful FPS
            time_span = self.recent_frames[-1] - self.recent_frames[0]
            fps = (len(self.recent_frames) - 1) / time_span if time_span > 1.0 else 0  # Require at least 1 second
        else:
            fps = 0
        
        # Calculate averages for key metrics (last 30 frames)
        avg_total = sum(self.stats['total_time'][-30:]) / min(30, len(self.stats['total_time'])) if self.stats['total_time'] else 0
        avg_caption = sum(self.stats['caption_time'][-30:]) / min(30, len(self.stats['caption_time'])) if self.stats['caption_time'] else 0
        avg_detection = sum(self.stats['detection_time'][-30:]) / min(30, len(self.stats['detection_time'])) if self.stats['detection_time'] else 0
        avg_segmentation = sum(self.stats['segmentation_time'][-30:]) / min(30, len(self.stats['segmentation_time'])) if self.stats['segmentation_time'] else 0
        avg_clip = sum(self.stats['clip_time'][-30:]) / min(30, len(self.stats['clip_time'])) if self.stats['clip_time'] else 0
        avg_mapping = sum(self.stats['mapping_time'][-30:]) / min(30, len(self.stats['mapping_time'])) if self.stats['mapping_time'] else 0
        
        print(f"\n📊 AveragePerformance Stats (Frame {self.frame_count}):")
        print(f"   FPS: {fps:.1f} | Total: {avg_total:.1f}ms")
        print(f"   Caption: {avg_caption:.1f}ms | Detection: {avg_detection:.1f}ms | Segmentation: {avg_segmentation:.1f}ms | CLIP: {avg_clip:.1f}ms | Mapping: {avg_mapping:.1f}ms")
        
        # Memory usage of recent data
        recent_data_size = len(self.frame_data)
        if recent_data_size > 1000:
            print(f"   📝 Logged {recent_data_size} frames")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self._data_lock:
            if not self.frame_data:
                return {}
            
            # Calculate FPS
            if len(self.recent_frames) > 1:
                time_span = self.recent_frames[-1] - self.recent_frames[0]
                fps = len(self.recent_frames) / time_span if time_span > 0 else 0
            else:
                fps = 0
            
            stats = {
                'frame_count': self.frame_count,
                'fps': fps,
                'uptime_seconds': time.time() - self.start_time,
            }
            
            # Add averages for key metrics
            for key, values in self.stats.items():
                if values:
                    recent_values = values[-50:]  # Last 50 frames
                    stats[f'{key}_avg'] = sum(recent_values) / len(recent_values)
                    stats[f'{key}_max'] = max(recent_values)
                    stats[f'{key}_min'] = min(recent_values)
            
            return stats
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        self.cleanup()
        exit(0)
    
    def _generate_summary_dict(self):
        """Generate summary statistics dictionary."""
        summary = {
            'session_info': {
                'total_frames': self.frame_count,
                'start_time': self.start_time,
                'end_time': time.time(),
                'duration_seconds': time.time() - self.start_time,
            },
            'performance_stats': {}
        }
        
        # Calculate comprehensive statistics
        for key, values in self.stats.items():
            if values:
                summary['performance_stats'][key] = {
                    'count': len(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'p50': sorted(values)[len(values)//2],
                    'p95': sorted(values)[int(len(values)*0.95)] if len(values) > 20 else max(values),
                    'p99': sorted(values)[int(len(values)*0.99)] if len(values) > 100 else max(values),
                }
        
        # Overall FPS calculation (only if meaningful duration)
        session_duration = time.time() - self.start_time
        if self.frame_count > 5 and session_duration > 5.0:  # At least 5 frames and 5 seconds
            overall_fps = self.frame_count / session_duration
            summary['session_info']['average_fps'] = overall_fps
        else:
            summary['session_info']['average_fps'] = 0.0  # Mark as invalid for short sessions
            
        return summary
    
    def write_scene_summary_and_reset(self, scene_name: str):
        """Write performance summary for current scene and reset data for next scene."""
        print(f"\n📊 Writing performance summary for scene: {scene_name}")
        
        with self._data_lock:
            if self.frame_count == 0:
                print(f"⚠️  No frames processed for scene {scene_name}, skipping summary")
                return
                
            # Generate summary statistics for this scene
            summary = self._generate_summary_dict()
            summary['session_info']['scene_name'] = scene_name
            
            # Write scene-specific summary
            scene_json_file = self.log_directory / f"performance_summary_{scene_name}.json"
            try:
                with open(scene_json_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"✅ Performance summary written to: {scene_json_file}")
                print(f"   📈 Scene {scene_name}: {self.frame_count} frames processed")
                
                # Also log FPS if available
                if 'average_fps' in summary['session_info'] and summary['session_info']['average_fps'] > 0:
                    fps = summary['session_info']['average_fps']
                    print(f"   🎯 Average FPS: {fps:.1f}")
                    
            except Exception as e:
                print(f"❌ Error writing scene summary: {e}")
            
            # Reset data for next scene
            self.frame_data.clear()
            self.recent_frames.clear()
            self.stats.clear()
            self.frame_count = 0
            self.start_time = time.time()
            
            print(f"🔄 Performance data reset for next scene")
    
    def cleanup(self):
        """Cleanup and write final summary."""
        if self._cleanup_done:
            return  # Already cleaned up
        
        self._cleanup_done = True
        
        with self._data_lock:
            # Generate summary statistics for console output
            summary = None
            
            # Only write final summary if there's remaining data
            if self.frame_count > 0:
                print("\n📊 Writing final performance summary for remaining data...")
                
                # Generate summary statistics  
                summary = self._generate_summary_dict()
                summary['session_info']['note'] = "Final cleanup - remaining data"
                
                # Write JSON summary
                try:
                    with open(self.json_file, 'w') as f:
                        json.dump(summary, f, indent=2)
                    print(f"📈 Final summary saved to: {self.json_file}")
                except Exception as e:
                    print(f"⚠️  Error writing final summary: {e}")
            else:
                print("\n📊 No remaining performance data to write (already written per-scene)")
            
            # Print final console summary
            print(f"\n🏁 Final Performance Summary:")
            print(f"   Total Frames: {self.frame_count}")
            print(f"   Session Duration: {time.time() - self.start_time:.1f}s")
            if summary and 'total_time' in summary['performance_stats']:
                avg_frame_time = summary['performance_stats']['total_time']['average']
                print(f"   Average Frame Time: {avg_frame_time:.1f}ms")
            if summary and 'average_fps' in summary['session_info']:
                print(f"   Average FPS: {summary['session_info']['average_fps']:.1f}")


# Global instance
_performance_manager = None

def get_performance_manager(scene_name: Optional[str] = None) -> PerformanceManager:
    """Get the global PerformanceManager instance."""
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = PerformanceManager(scene_name=scene_name)
    return _performance_manager


# Convenience functions
def log_frame_timing(frame_number: int, timing_dict: Dict[str, float], **kwargs):
    """Convenience function to log frame timing."""
    get_performance_manager().log_frame_timing(frame_number, timing_dict, **kwargs)

def get_current_stats() -> Dict[str, Any]:
    """Convenience function to get current stats."""
    return get_performance_manager().get_current_stats()