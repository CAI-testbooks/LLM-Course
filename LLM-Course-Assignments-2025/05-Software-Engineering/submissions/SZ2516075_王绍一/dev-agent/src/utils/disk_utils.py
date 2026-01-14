# src/utils/disk_utils.py
import os
import shutil
from typing import Tuple, List

class DiskUtils:
    """磁盘工具类"""
    
    @staticmethod
    def get_disk_space(path: str) -> Tuple[float, float, float]:
        """获取磁盘空间信息 (GB)"""
        try:
            total, used, free = shutil.disk_usage(path)
            return (
                total / (1024**3),  # GB
                used / (1024**3),   # GB
                free / (1024**3)    # GB
            )
        except Exception as e:
            print(f"无法获取磁盘空间: {e}")
            return 0, 0, 0
    
    @staticmethod
    def get_best_drive_for_cache(min_free_gb: float = 5) -> str:
        """获取最适合做缓存的磁盘"""
        # Windows磁盘
        drives = ["D:", "E:", "C:"]
        
        for drive in drives:
            try:
                total, used, free = DiskUtils.get_disk_space(drive)
                if free >= min_free_gb:
                    return drive
            except:
                continue
        
        # 如果没有找到合适的磁盘，返回当前目录的磁盘
        return os.path.splitdrive(os.getcwd())[0]
    
    @staticmethod
    def get_cache_drives_with_space() -> List[Tuple[str, float]]:
        """获取所有有空间的缓存磁盘"""
        drives = []
        for drive in ["D:", "E:", "C:"]:
            try:
                total, used, free = DiskUtils.get_disk_space(drive)
                if free > 1:  # 至少1GB空间
                    drives.append((drive, free))
            except:
                continue
        return sorted(drives, key=lambda x: x[1], reverse=True)  # 按空间大小排序
    
    @staticmethod
    def cleanup_cache(cache_dir: str, keep_gb: float = 5):
        """清理缓存，保留指定大小的空间"""
        if not os.path.exists(cache_dir):
            return
        
        # 获取缓存大小
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
        
        cache_size_gb = total_size / (1024**3)
        
        if cache_size_gb > keep_gb:
            print(f"🧹 清理缓存: {cache_size_gb:.1f}GB > {keep_gb}GB")
            # 这里可以添加具体的清理逻辑
            # 例如删除最旧的文件等