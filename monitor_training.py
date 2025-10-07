#!/usr/bin/env python3
import time
import subprocess
import re

def get_training_progress():
    """获取训练进度信息"""
    try:
        # 查找训练进程
        result = subprocess.run(['pgrep', '-f', 'train_denoiser.py'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            return "训练进程未找到"
        
        pid = result.stdout.strip().split('\n')[0]
        
        # 获取进程的标准输出（如果可能）
        # 这里我们简单地检查GPU使用情况
        gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
        
        if gpu_result.returncode == 0:
            gpu_info = gpu_result.stdout.strip()
            return f"训练进程PID: {pid}, GPU状态: {gpu_info}"
        else:
            return f"训练进程PID: {pid}, GPU信息获取失败"
            
    except Exception as e:
        return f"监控失败: {e}"

def main():
    print("开始监控训练进度...")
    print("=" * 50)
    
    for i in range(10):  # 监控10次，每次间隔30秒
        timestamp = time.strftime("%H:%M:%S")
        progress = get_training_progress()
        print(f"[{timestamp}] {progress}")
        
        if i < 9:  # 最后一次不需要等待
            time.sleep(30)
    
    print("=" * 50)
    print("监控结束")

if __name__ == "__main__":
    main()
