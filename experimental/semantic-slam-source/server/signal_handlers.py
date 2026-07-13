# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Signal handling for graceful shutdown of SLAM server.

This module provides signal handlers for clean termination of all processes
and proper cleanup of resources.
"""

import signal
import sys
import time
from typing import List, Optional
import multiprocessing


class GracefulShutdownHandler:
    """Handles graceful shutdown of SLAM server processes."""
    
    def __init__(self):
        self.shutdown_requested = False
        self.processes: List[multiprocessing.Process] = []
        self.performance_manager = None
        
    def register_processes(self, processes: List[multiprocessing.Process]):
        """Register processes for shutdown management."""
        self.processes = processes
        
    def register_performance_manager(self, perf_manager):
        """Register performance manager for final logging."""
        self.performance_manager = perf_manager
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        if self.shutdown_requested:
            print("\n⚠️  Force quit detected, terminating immediately...")
            sys.exit(1)
            
        self.shutdown_requested = True
        signal_name = signal.Signals(signum).name
        print(f"\n🛑 Received {signal_name} signal, initiating graceful shutdown...")
        
        self.shutdown()
        
    def shutdown(self):
        """Perform graceful shutdown sequence."""
        print("📊 Finalizing performance logs...")
        
        # Give performance manager time to finish current operations
        if self.performance_manager:
            try:
                self.performance_manager.cleanup()
            except Exception as e:
                print(f"⚠️  Error during performance manager cleanup: {e}")
        
        print("🔄 Terminating SLAM processes...")
        
        # Terminate processes gracefully
        for i, process in enumerate(self.processes):
            if process and process.is_alive():
                print(f"   Stopping process {i+1}/{len(self.processes)}: {process.name or 'Unknown'}")
                try:
                    process.terminate()
                    process.join(timeout=5.0)  # Wait up to 5 seconds
                    
                    if process.is_alive():
                        print(f"   Force killing process {i+1}")
                        process.kill()
                        process.join(timeout=2.0)
                        
                except Exception as e:
                    print(f"   Error stopping process {i+1}: {e}")
        
        print("✅ Graceful shutdown completed")
        sys.exit(0)
    
    def install_handlers(self):
        """Install signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, self.signal_handler)  # Termination request
        
        # On Unix systems, also handle SIGUSR1 for custom shutdown
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self.signal_handler)
            
        # print("🛡️  Signal handlers installed for graceful shutdown")
        # print("   Press Ctrl+C to initiate graceful shutdown")


# Global instance for convenience
_shutdown_handler: Optional[GracefulShutdownHandler] = None

def get_shutdown_handler() -> GracefulShutdownHandler:
    """Get the global shutdown handler instance."""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdownHandler()
    return _shutdown_handler

def install_signal_handlers():
    """Convenience function to install signal handlers."""
    handler = get_shutdown_handler()
    handler.install_handlers()
    return handler