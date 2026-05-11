"""
GPU detection and configuration utilities for accelerated model training.

Detects available GPUs and provides device selection for XGBoost, LightGBM, CuPy, etc.
"""

import os
import subprocess
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """GPU device information."""
    device_id: int
    name: str
    compute_capability: str
    total_memory_gb: float
    is_available: bool = True

    def __str__(self) -> str:
        return f"GPU {self.device_id}: {self.name} ({self.compute_capability}) - {self.total_memory_gb:.1f}GB"


class GPUDetector:
    """Detect and manage available GPU devices."""
    
    @staticmethod
    def detect_gpu_availability() -> Tuple[bool, Optional[str]]:
        """
        Detect if GPU is available and return device type.
        
        Returns:
            Tuple of (is_available: bool, device_type: Optional[str])
            device_type can be: 'cuda', 'rocm', 'mps' (Apple Metal), None
        """
        # Check for CUDA (NVIDIA)
        if GPUDetector._check_cuda_available():
            return True, 'cuda'
        
        # Check for ROCm (AMD)
        if GPUDetector._check_rocm_available():
            return True, 'rocm'
        
        # Check for MPS (Apple Metal Performance Shaders)
        if GPUDetector._check_mps_available():
            return True, 'mps'
        
        return False, None
    
    @staticmethod
    def _check_cuda_available() -> bool:
        """Check if NVIDIA CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            import cupy
            return cupy.cuda.is_available()
        except ImportError:
            pass
        
        # Fallback: check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0 and int(result.stdout.strip().split('\n')[0]) > 0
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            return False
    
    @staticmethod
    def _check_rocm_available() -> bool:
        """Check if AMD ROCm is available."""
        try:
            import torch
            return hasattr(torch.version, 'hip') and torch.version.hip is not None
        except ImportError:
            pass
        
        try:
            subprocess.run(['rocminfo'], capture_output=True, timeout=5, check=True)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return False
    
    @staticmethod
    def _check_mps_available() -> bool:
        """Check if Apple Metal Performance Shaders is available."""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            return False
    
    @staticmethod
    def list_gpu_devices(device_type: str = 'cuda') -> List[GPUInfo]:
        """
        List available GPU devices.
        
        Args:
            device_type: 'cuda', 'rocm', or 'mps'
            
        Returns:
            List of GPUInfo objects
        """
        devices = []
        
        if device_type == 'cuda':
            try:
                import torch
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    cap = torch.cuda.get_device_capability(i)
                    compute_capability = f"{cap[0]}.{cap[1]}"
                    
                    # Get memory
                    props = torch.cuda.get_device_properties(i)
                    total_memory_gb = props.total_memory / (1024 ** 3)
                    
                    devices.append(GPUInfo(
                        device_id=i,
                        name=name,
                        compute_capability=compute_capability,
                        total_memory_gb=total_memory_gb
                    ))
                return devices
            except ImportError:
                pass
        
        # Try nvidia-smi as fallback
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            device_id = int(parts[0])
                            name = parts[1]
                            memory_str = parts[2].replace(' MiB', '')
                            compute_cap = parts[3] if len(parts) > 3 else 'unknown'
                            
                            try:
                                total_memory_gb = float(memory_str) / 1024
                            except ValueError:
                                total_memory_gb = 0.0
                            
                            devices.append(GPUInfo(
                                device_id=device_id,
                                name=name,
                                compute_capability=compute_cap,
                                total_memory_gb=total_memory_gb
                            ))
                return devices
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        
        return devices
    
    @staticmethod
    def get_best_gpu() -> Optional[GPUInfo]:
        """Get the GPU with most memory."""
        is_available, device_type = GPUDetector.detect_gpu_availability()
        if not is_available or device_type is None:
            return None
        
        devices = GPUDetector.list_gpu_devices(device_type)
        if not devices:
            return None
        
        return max(devices, key=lambda d: d.total_memory_gb)
    
    @staticmethod
    def get_device_config() -> Dict[str, any]:
        """Get configuration dict for ML frameworks."""
        is_available, device_type = GPUDetector.detect_gpu_availability()
        
        config = {
            'gpu_available': is_available,
            'device_type': device_type,
            'devices': [],
        }
        
        if is_available and device_type:
            config['devices'] = [
                {
                    'id': d.device_id,
                    'name': d.name,
                    'compute_capability': d.compute_capability,
                    'memory_gb': d.total_memory_gb
                }
                for d in GPUDetector.list_gpu_devices(device_type)
            ]
        
        return config


class GPUTrainerConfig:
    """Configuration for GPU-accelerated training."""
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize GPU trainer config.
        
        Args:
            device: 'auto' (detect), 'cuda', 'rocm', 'mps', 'cpu'
        """
        self.device = device
        self.is_gpu_available = False
        self.actual_device = 'cpu'
        
        if device == 'auto':
            is_available, detected_device = GPUDetector.detect_gpu_availability()
            self.is_gpu_available = is_available
            if is_available and detected_device:
                self.actual_device = detected_device
        elif device in ('cuda', 'rocm', 'mps'):
            is_available, _ = GPUDetector.detect_gpu_availability()
            self.is_gpu_available = is_available and device == 'auto' or \
                                   GPUDetector._check_cuda_available() and device == 'cuda' or \
                                   GPUDetector._check_rocm_available() and device == 'rocm' or \
                                   GPUDetector._check_mps_available() and device == 'mps'
            if self.is_gpu_available:
                self.actual_device = device
    
    def get_xgboost_params(self) -> Dict:
        """Get XGBoost GPU parameters."""
        if not self.is_gpu_available:
            return {'tree_method': 'hist'}  # CPU fallback
        
        if self.actual_device == 'cuda':
            return {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'device': 'cuda'
            }
        elif self.actual_device == 'rocm':
            return {
                'tree_method': 'gpu_hist',
                'device': 'gpu'
            }
        else:
            return {'tree_method': 'hist'}
    
    def get_lightgbm_params(self) -> Dict:
        """Get LightGBM GPU parameters."""
        if not self.is_gpu_available:
            return {}  # CPU is default
        
        if self.actual_device == 'cuda':
            return {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            }
        elif self.actual_device == 'rocm':
            return {
                'device': 'gpu'
            }
        else:
            return {}
    
    def __str__(self) -> str:
        status = f"GPU Device: {self.actual_device.upper()}"
        if not self.is_gpu_available:
            status = "GPU Device: CPU (fallback)"
        return status


def print_gpu_info():
    """Print GPU detection information."""
    is_available, device_type = GPUDetector.detect_gpu_availability()
    
    print("\n" + "=" * 80)
    print("GPU Detection Report")
    print("=" * 80)
    
    if is_available and device_type:
        print(f"✅ GPU Available: {device_type.upper()}")
        devices = GPUDetector.list_gpu_devices(device_type)
        for device in devices:
            print(f"   {device}")
        
        best = GPUDetector.get_best_gpu()
        if best:
            print(f"\n🎯 Recommended GPU: GPU {best.device_id} ({best.name})")
    else:
        print("❌ No GPU detected - will use CPU")
        print("   Install NVIDIA CUDA or AMD ROCm for GPU acceleration")
    
    print("=" * 80 + "\n")


if __name__ == '__main__':
    print_gpu_info()
