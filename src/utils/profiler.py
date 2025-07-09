# File: src/utils/profiler.py

"""
Performance monitoring and profiling utilities for the BI-Net project.
Provides timing, memory usage tracking, and performance profiling capabilities.
"""

import time
import torch
import psutil
import functools
import threading
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.execution_times = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.gpu_memory_usage = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.peak_memory = 0
        self.peak_gpu_memory = 0
        
    def add_timing(self, name: str, duration: float):
        """Add timing measurement."""
        self.execution_times[name].append(duration)
        self.call_counts[name] += 1
        
    def add_memory(self, name: str, memory_mb: float):
        """Add memory usage measurement."""
        self.memory_usage[name].append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
        
    def add_gpu_memory(self, name: str, memory_mb: float):
        """Add GPU memory usage measurement."""
        self.gpu_memory_usage[name].append(memory_mb)
        self.peak_gpu_memory = max(self.peak_gpu_memory, memory_mb)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'execution_times': {},
            'memory_usage': {},
            'gpu_memory_usage': {},
            'call_counts': dict(self.call_counts),
            'peak_memory_mb': self.peak_memory,
            'peak_gpu_memory_mb': self.peak_gpu_memory
        }
        
        # Calculate timing statistics
        for name, times in self.execution_times.items():
            if times:
                stats['execution_times'][name] = {
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times),
                    'count': len(times)
                }
        
        # Calculate memory statistics
        for name, memories in self.memory_usage.items():
            if memories:
                stats['memory_usage'][name] = {
                    'mean': sum(memories) / len(memories),
                    'min': min(memories),
                    'max': max(memories),
                    'count': len(memories)
                }
        
        # Calculate GPU memory statistics
        for name, memories in self.gpu_memory_usage.items():
            if memories:
                stats['gpu_memory_usage'][name] = {
                    'mean': sum(memories) / len(memories),
                    'min': min(memories),
                    'max': max(memories),
                    'count': len(memories)
                }
        
        return stats


class PerformanceProfiler:
    """Main performance profiler class."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = PerformanceMetrics()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._memory_history = deque(maxlen=1000)
        
    def enable(self):
        """Enable performance profiling."""
        self.enabled = True
        
    def disable(self):
        """Disable performance profiling."""
        self.enabled = False
        
    def reset(self):
        """Reset all performance metrics."""
        self.metrics.reset()
        self._memory_history.clear()
        
    @contextmanager
    def profile_context(self, name: str, track_memory: bool = True):
        """Context manager for profiling a code block."""
        if not self.enabled:
            yield
            return
            
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage() if track_memory else 0
        start_gpu_memory = self._get_gpu_memory_usage() if track_memory else 0
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            self.metrics.add_timing(name, duration)
            
            if track_memory:
                end_memory = self._get_memory_usage()
                end_gpu_memory = self._get_gpu_memory_usage()
                
                self.metrics.add_memory(name, end_memory)
                self.metrics.add_gpu_memory(name, end_gpu_memory)
                
                # Track memory usage delta
                memory_delta = end_memory - start_memory
                if memory_delta > 0:
                    logger.debug(f"Memory increase in {name}: {memory_delta:.2f} MB")
    
    def profile_function(self, name: Optional[str] = None, track_memory: bool = True):
        """Decorator for profiling functions."""
        def decorator(func: Callable) -> Callable:
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_context(func_name, track_memory):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def start_continuous_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring in a separate thread."""
        if not self.enabled or self._monitoring_thread is not None:
            return
            
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_memory,
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        
    def stop_continuous_monitoring(self):
        """Stop continuous memory monitoring."""
        if self._monitoring_thread is not None:
            self._stop_monitoring.set()
            self._monitoring_thread.join()
            self._monitoring_thread = None
    
    def _monitor_memory(self, interval: float):
        """Memory monitoring thread function."""
        while not self._stop_monitoring.wait(interval):
            memory_mb = self._get_memory_usage()
            gpu_memory_mb = self._get_gpu_memory_usage()
            
            self._memory_history.append({
                'timestamp': time.time(),
                'memory_mb': memory_mb,
                'gpu_memory_mb': gpu_memory_mb
            })
            
            self.metrics.add_memory('continuous_monitoring', memory_mb)
            self.metrics.add_gpu_memory('continuous_monitoring', gpu_memory_mb)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            return 0.0
        except Exception:
            return 0.0
    
    def get_memory_history(self) -> List[Dict[str, Any]]:
        """Get memory usage history."""
        return list(self._memory_history)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.metrics.get_stats()
        stats['memory_history'] = self.get_memory_history()
        return stats
    
    def print_stats(self):
        """Print performance statistics to console."""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("PERFORMANCE PROFILING REPORT")
        print("="*50)
        
        # Execution times
        if stats['execution_times']:
            print("\nExecution Times:")
            print("-" * 30)
            for name, time_stats in stats['execution_times'].items():
                print(f"{name:30} | "
                      f"Mean: {time_stats['mean']:.4f}s | "
                      f"Total: {time_stats['total']:.4f}s | "
                      f"Calls: {time_stats['count']}")
        
        # Memory usage
        if stats['memory_usage']:
            print("\nMemory Usage:")
            print("-" * 30)
            for name, mem_stats in stats['memory_usage'].items():
                print(f"{name:30} | "
                      f"Mean: {mem_stats['mean']:.2f}MB | "
                      f"Peak: {mem_stats['max']:.2f}MB")
        
        # GPU memory usage
        if stats['gpu_memory_usage']:
            print("\nGPU Memory Usage:")
            print("-" * 30)
            for name, mem_stats in stats['gpu_memory_usage'].items():
                print(f"{name:30} | "
                      f"Mean: {mem_stats['mean']:.2f}MB | "
                      f"Peak: {mem_stats['max']:.2f}MB")
        
        # Overall peaks
        print(f"\nOverall Peak Memory: {stats['peak_memory_mb']:.2f}MB")
        print(f"Overall Peak GPU Memory: {stats['peak_gpu_memory_mb']:.2f}MB")
        
        # Most called functions
        if stats['call_counts']:
            print("\nMost Called Functions:")
            print("-" * 30)
            sorted_calls = sorted(stats['call_counts'].items(), 
                                key=lambda x: x[1], reverse=True)
            for name, count in sorted_calls[:10]:
                print(f"{name:30} | Calls: {count}")
        
        print("="*50)
    
    def save_stats(self, filename: str):
        """Save performance statistics to JSON file."""
        stats = self.get_stats()
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Performance stats saved to {filename}")
    
    def profile_pytorch_model(self, model: torch.nn.Module, 
                            input_data: torch.Tensor, 
                            name: str = "model_forward"):
        """Profile PyTorch model execution."""
        if not self.enabled:
            return
            
        model.eval()
        with torch.no_grad():
            # Warmup
            _ = model(input_data)
            
            # Profile
            with self.profile_context(name):
                output = model(input_data)
                
        return output


# Global profiler instance
_global_profiler = PerformanceProfiler(enabled=False)

def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler

def enable_profiling():
    """Enable global profiling."""
    _global_profiler.enable()

def disable_profiling():
    """Disable global profiling."""
    _global_profiler.disable()

def profile_function(name: Optional[str] = None, track_memory: bool = True):
    """Decorator for profiling functions using global profiler."""
    return _global_profiler.profile_function(name, track_memory)

@contextmanager
def profile_context(name: str, track_memory: bool = True):
    """Context manager for profiling using global profiler."""
    with _global_profiler.profile_context(name, track_memory):
        yield

def print_profiling_stats():
    """Print profiling statistics using global profiler."""
    _global_profiler.print_stats()

def save_profiling_stats(filename: str):
    """Save profiling statistics using global profiler."""
    _global_profiler.save_stats(filename)

def reset_profiling():
    """Reset profiling statistics using global profiler."""
    _global_profiler.reset()


# Specialized profilers for common operations
class TrainingProfiler:
    """Specialized profiler for training operations."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.epoch_stats = []
        
    def start_epoch(self):
        """Start profiling an epoch."""
        self.profiler.reset()
        self.profiler.start_continuous_monitoring(interval=5.0)
        
    def end_epoch(self, epoch: int):
        """End profiling an epoch."""
        self.profiler.stop_continuous_monitoring()
        stats = self.profiler.get_stats()
        stats['epoch'] = epoch
        self.epoch_stats.append(stats)
        
    def profile_forward_pass(self, model: torch.nn.Module, 
                           input_data: torch.Tensor):
        """Profile forward pass."""
        with self.profiler.profile_context("forward_pass"):
            return model(input_data)
            
    def profile_backward_pass(self, loss: torch.Tensor):
        """Profile backward pass."""
        with self.profiler.profile_context("backward_pass"):
            loss.backward()
            
    def profile_optimizer_step(self, optimizer: torch.optim.Optimizer):
        """Profile optimizer step."""
        with self.profiler.profile_context("optimizer_step"):
            optimizer.step()
            
    def get_epoch_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all epochs."""
        return self.epoch_stats
    
    def save_epoch_stats(self, filename: str):
        """Save epoch statistics to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.epoch_stats, f, indent=2)
        logger.info(f"Epoch stats saved to {filename}")


class DataLoaderProfiler:
    """Specialized profiler for data loading operations."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.batch_times = []
        
    def profile_batch_loading(self, dataloader):
        """Profile batch loading from dataloader."""
        for i, batch in enumerate(dataloader):
            with self.profiler.profile_context(f"batch_{i}"):
                # Data is already loaded at this point
                pass
                
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch loading statistics."""
        return self.profiler.get_stats()