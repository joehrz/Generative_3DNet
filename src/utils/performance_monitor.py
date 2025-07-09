# File: src/utils/performance_monitor.py

"""
Performance monitoring integration for the BI-Net training pipeline.
Provides comprehensive monitoring of training performance, memory usage, and bottlenecks.
"""

import time
import torch
import json
import os
from typing import Dict, Any, Optional
from src.utils.profiler import TrainingProfiler, get_profiler, enable_profiling
from src.utils.exceptions import PerformanceError
import logging

logger = logging.getLogger(__name__)


class BiNetPerformanceMonitor:
    """
    Comprehensive performance monitor for BI-Net training pipeline.
    Tracks timing, memory usage, and identifies performance bottlenecks.
    """
    
    def __init__(self, config, save_dir: str = "performance_logs"):
        self.config = config
        self.save_dir = save_dir
        self.training_profiler = TrainingProfiler()
        self.session_start_time = time.time()
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Enable global profiling
        enable_profiling()
        
        # Performance thresholds (configurable)
        self.thresholds = {
            'max_memory_mb': 16000,  # 16GB
            'max_gpu_memory_mb': 8000,  # 8GB
            'max_batch_time_sec': 60,  # 1 minute per batch
            'max_epoch_time_sec': 3600,  # 1 hour per epoch
            'memory_leak_threshold_mb': 100,  # 100MB increase per epoch
        }
        
        # Performance tracking
        self.epoch_times = []
        self.memory_usage_trend = []
        self.bottleneck_analysis = {}
        
    def start_training_session(self):
        """Start monitoring a complete training session."""
        logger.info("Starting performance monitoring for training session")
        self.session_start_time = time.time()
        
        # Log system information
        self._log_system_info()
        
    def start_epoch(self, epoch: int):
        """Start monitoring an epoch."""
        logger.info(f"Starting performance monitoring for epoch {epoch}")
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.training_profiler.start_epoch()
        
    def end_epoch(self, epoch: int):
        """End monitoring an epoch and analyze performance."""
        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_duration)
        
        # End profiling
        self.training_profiler.end_epoch(epoch)
        
        # Analyze performance
        self._analyze_epoch_performance(epoch, epoch_duration)
        
        # Check for issues
        self._check_performance_issues(epoch)
        
        # Save epoch stats
        self._save_epoch_stats(epoch)
        
        logger.info(f"Epoch {epoch} completed in {epoch_duration:.2f}s")
        
    def monitor_forward_pass(self, model: torch.nn.Module, 
                           input_data: torch.Tensor, 
                           pass_type: str = "forward"):
        """Monitor a forward pass."""
        return self.training_profiler.profile_forward_pass(model, input_data)
        
    def monitor_backward_pass(self, loss: torch.Tensor):
        """Monitor a backward pass."""
        self.training_profiler.profile_backward_pass(loss)
        
    def monitor_optimizer_step(self, optimizer: torch.optim.Optimizer):
        """Monitor an optimizer step."""
        self.training_profiler.profile_optimizer_step(optimizer)
        
    def monitor_data_loading(self, dataloader, name: str = "train_loader"):
        """Monitor data loading performance."""
        loading_times = []
        
        for i, batch in enumerate(dataloader):
            start_time = time.time()
            # Data is already loaded
            loading_time = time.time() - start_time
            loading_times.append(loading_time)
            
            if i >= 10:  # Sample first 10 batches
                break
                
        avg_loading_time = sum(loading_times) / len(loading_times)
        logger.info(f"{name} average loading time: {avg_loading_time:.4f}s")
        
        return avg_loading_time
        
    def _log_system_info(self):
        """Log system information for performance context."""
        system_info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'batch_size': self.config.training.batch_size,
            'learning_rate': self.config.training.learning_rate,
            'model_latent_dim': self.config.model.latent_dim,
        }
        
        logger.info("System Information:")
        for key, value in system_info.items():
            logger.info(f"  {key}: {value}")
            
        # Save system info
        with open(os.path.join(self.save_dir, "system_info.json"), 'w') as f:
            json.dump(system_info, f, indent=2)
            
    def _analyze_epoch_performance(self, epoch: int, duration: float):
        """Analyze performance for the completed epoch."""
        profiler = get_profiler()
        stats = profiler.get_stats()
        
        # Analyze timing bottlenecks
        if stats['execution_times']:
            slowest_operations = sorted(
                stats['execution_times'].items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )[:5]
            
            self.bottleneck_analysis[epoch] = {
                'slowest_operations': slowest_operations,
                'total_epoch_time': duration,
                'forward_pass_time': stats['execution_times'].get('forward_pass', {}).get('total', 0),
                'backward_pass_time': stats['execution_times'].get('backward_pass', {}).get('total', 0),
                'optimizer_step_time': stats['execution_times'].get('optimizer_step', {}).get('total', 0),
            }
            
        # Track memory usage trend
        current_memory = stats.get('peak_memory_mb', 0)
        self.memory_usage_trend.append({
            'epoch': epoch,
            'memory_mb': current_memory,
            'gpu_memory_mb': stats.get('peak_gpu_memory_mb', 0)
        })
        
    def _check_performance_issues(self, epoch: int):
        """Check for performance issues and log warnings."""
        profiler = get_profiler()
        stats = profiler.get_stats()
        
        # Check memory usage
        current_memory = stats.get('peak_memory_mb', 0)
        current_gpu_memory = stats.get('peak_gpu_memory_mb', 0)
        
        if current_memory > self.thresholds['max_memory_mb']:
            logger.warning(f"High memory usage detected: {current_memory:.2f}MB "
                          f"(threshold: {self.thresholds['max_memory_mb']}MB)")
                          
        if current_gpu_memory > self.thresholds['max_gpu_memory_mb']:
            logger.warning(f"High GPU memory usage detected: {current_gpu_memory:.2f}MB "
                          f"(threshold: {self.thresholds['max_gpu_memory_mb']}MB)")
        
        # Check for memory leaks
        if len(self.memory_usage_trend) > 1:
            memory_increase = (current_memory - 
                             self.memory_usage_trend[-2]['memory_mb'])
            if memory_increase > self.thresholds['memory_leak_threshold_mb']:
                logger.warning(f"Potential memory leak detected: "
                              f"{memory_increase:.2f}MB increase from last epoch")
        
        # Check epoch duration
        if len(self.epoch_times) > 0:
            last_epoch_time = self.epoch_times[-1]
            if last_epoch_time > self.thresholds['max_epoch_time_sec']:
                logger.warning(f"Slow epoch detected: {last_epoch_time:.2f}s "
                              f"(threshold: {self.thresholds['max_epoch_time_sec']}s)")
                              
    def _save_epoch_stats(self, epoch: int):
        """Save detailed epoch statistics."""
        profiler = get_profiler()
        stats = profiler.get_stats()
        
        epoch_stats = {
            'epoch': epoch,
            'duration': self.epoch_times[-1] if self.epoch_times else 0,
            'performance_stats': stats,
            'bottleneck_analysis': self.bottleneck_analysis.get(epoch, {}),
            'memory_trend': self.memory_usage_trend[-10:],  # Last 10 epochs
        }
        
        # Save to file
        stats_file = os.path.join(self.save_dir, f"epoch_{epoch:04d}_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(epoch_stats, f, indent=2)
            
    def generate_performance_report(self, output_file: str = None):
        """Generate a comprehensive performance report."""
        if output_file is None:
            output_file = os.path.join(self.save_dir, "performance_report.json")
            
        total_training_time = time.time() - self.session_start_time
        
        report = {
            'summary': {
                'total_training_time': total_training_time,
                'total_epochs': len(self.epoch_times),
                'average_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
                'fastest_epoch_time': min(self.epoch_times) if self.epoch_times else 0,
                'slowest_epoch_time': max(self.epoch_times) if self.epoch_times else 0,
            },
            'memory_analysis': {
                'peak_memory_mb': max(m['memory_mb'] for m in self.memory_usage_trend) if self.memory_usage_trend else 0,
                'peak_gpu_memory_mb': max(m['gpu_memory_mb'] for m in self.memory_usage_trend) if self.memory_usage_trend else 0,
                'memory_trend': self.memory_usage_trend,
            },
            'bottleneck_analysis': self.bottleneck_analysis,
            'recommendations': self._generate_recommendations(),
            'epoch_times': self.epoch_times,
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Performance report saved to {output_file}")
        
        # Print summary
        self._print_performance_summary(report)
        
        return report
        
    def _generate_recommendations(self) -> list:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze bottlenecks
        if self.bottleneck_analysis:
            all_bottlenecks = {}
            for epoch_bottlenecks in self.bottleneck_analysis.values():
                for op_name, op_stats in epoch_bottlenecks.get('slowest_operations', []):
                    if op_name not in all_bottlenecks:
                        all_bottlenecks[op_name] = []
                    all_bottlenecks[op_name].append(op_stats['total'])
            
            # Find consistently slow operations
            for op_name, times in all_bottlenecks.items():
                avg_time = sum(times) / len(times)
                if avg_time > 1.0:  # Operations taking > 1 second on average
                    recommendations.append(f"Optimize '{op_name}' operation (avg: {avg_time:.2f}s)")
        
        # Memory recommendations
        if self.memory_usage_trend:
            peak_memory = max(m['memory_mb'] for m in self.memory_usage_trend)
            if peak_memory > 8000:  # > 8GB
                recommendations.append("Consider reducing batch size to lower memory usage")
            
            # Check for memory growth
            if len(self.memory_usage_trend) > 5:
                early_memory = sum(m['memory_mb'] for m in self.memory_usage_trend[:5]) / 5
                late_memory = sum(m['memory_mb'] for m in self.memory_usage_trend[-5:]) / 5
                if late_memory > early_memory * 1.2:  # 20% increase
                    recommendations.append("Potential memory leak detected - check for unreleased tensors")
        
        # Epoch time recommendations
        if self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            if avg_epoch_time > 300:  # > 5 minutes
                recommendations.append("Consider data loading optimizations or increasing num_workers")
        
        return recommendations
        
    def _print_performance_summary(self, report: Dict[str, Any]):
        """Print a summary of the performance report."""
        print("\n" + "="*60)
        print("TRAINING PERFORMANCE SUMMARY")
        print("="*60)
        
        summary = report['summary']
        print(f"Total Training Time: {summary['total_training_time']:.2f}s")
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Average Epoch Time: {summary['average_epoch_time']:.2f}s")
        print(f"Fastest Epoch: {summary['fastest_epoch_time']:.2f}s")
        print(f"Slowest Epoch: {summary['slowest_epoch_time']:.2f}s")
        
        memory = report['memory_analysis']
        print(f"\nPeak Memory Usage: {memory['peak_memory_mb']:.2f}MB")
        print(f"Peak GPU Memory Usage: {memory['peak_gpu_memory_mb']:.2f}MB")
        
        recommendations = report['recommendations']
        if recommendations:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        print("="*60)
        
    def cleanup(self):
        """Cleanup performance monitoring resources."""
        self.training_profiler.profiler.stop_continuous_monitoring()
        
    def __enter__(self):
        """Context manager entry."""
        self.start_training_session()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        if exc_type is None:
            self.generate_performance_report()