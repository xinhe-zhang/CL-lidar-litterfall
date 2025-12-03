import cupy as cp
import subprocess

print("=" * 60)
print("GPU 資訊檢查")
print("=" * 60)

try:
    n_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"\n可用 GPU 數量: {n_gpus}")
    
    for i in range(n_gpus):
        device = cp.cuda.Device(i)
        device.use()
        
        props = cp.cuda.runtime.getDeviceProperties(i)
        free_mem, total_mem = device.mem_info
        
        print(f"\n{'='*60}")
        print(f"GPU {i}: {props['name'].decode('utf-8')}")
        print(f"{'='*60}")
        print(f"總記憶體:        {total_mem / 1024**3:.2f} GB")
        print(f"可用記憶體:      {free_mem / 1024**3:.2f} GB")
        print(f"已使用記憶體:    {(total_mem - free_mem) / 1024**3:.2f} GB")
        print(f"使用率:          {(1 - free_mem/total_mem) * 100:.1f}%")
        print(f"\nCUDA 核心數:     {props['multiProcessorCount']}")
        print(f"計算能力:        {props['major']}.{props['minor']}")
        print(f"最大執行緒/區塊: {props['maxThreadsPerBlock']}")

except Exception as e:
    print(f"\n使用 CuPy 查詢失敗: {e}")
    print("\n嘗試使用 nvidia-smi...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e2:
        print(f"nvidia-smi 也失敗: {e2}")

print("\n" + "=" * 60)
print("建議:")
print("=" * 60)

try:
    device = cp.cuda.Device(0)
    free_mem, total_mem = device.mem_info
    free_gb = free_mem / 1024**3
    
    if free_gb < 2:
        print("警告: 可用 GPU 記憶體不足 2 GB")
        print("建議: CHUNK_SIZE = 10000, POSTERIOR_BATCH_SIZE = 2000")
    elif free_gb < 4:
        print("GPU 記憶體適中 (2-4 GB)")
        print("建議: CHUNK_SIZE = 20000, POSTERIOR_BATCH_SIZE = 5000")
    elif free_gb < 8:
        print("GPU 記憶體充足 (4-8 GB)")
        print("建議: CHUNK_SIZE = 40000, POSTERIOR_BATCH_SIZE = 8000")
    else:
        print(f"GPU 記憶體充裕 ({free_gb:.1f} GB)")
        print("建議: CHUNK_SIZE = 50000, POSTERIOR_BATCH_SIZE = 10000")
except:
    pass

print("=" * 60)