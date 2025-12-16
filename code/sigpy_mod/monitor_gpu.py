import time
from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex, nvmlShutdown, nvmlDeviceGetUtilizationRates
import threading

def monitor_gpu_memory(gpu_index=0, interval=0.1):
    """
    Monitors GPU memory usage every `interval` seconds between tic and toc.
    
    Args:
        gpu_index (int): The GPU index to monitor (default is 0).
        interval (int): Time in seconds between memory usage samples.

    Returns:
        tuple: (tic function, toc function, memory usage array).
    """
    # Initialize NVIDIA Management Library
    nvmlInit()
            
    # Sometimes, the GPU index being read might be mismatched from the expected GPU index being requested, for example:
    # if gpu_index == 5: 
    #     gpu_index = 2
    
    # Get handle for the specified GPU
    handle = nvmlDeviceGetHandleByIndex(gpu_index)
    
    memory_usage = []
    monitoring = False
    monitor_thread = None  # Declare monitor_thread globally for access in toc

    def tic():
        nonlocal monitoring, monitor_thread
        monitoring = True
        memory_usage.clear()  # Reset memory usage log
        
        # Start the monitoring thread
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
    
    def toc():
        nonlocal monitoring
        monitoring = False
        
        # Ensure the monitoring thread finishes
        if monitor_thread is not None:
            monitor_thread.join()
    
    # Monitoring function that will run in a separate thread
    def monitor():
        while monitoring:
            # Get memory info and store the used memory in MB
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            util_info = nvmlDeviceGetUtilizationRates(handle)
            memory_usage.append(mem_info.used / (1024 ** 2)) # Convert to MB
            time.sleep(interval)

    # Shutdown NVIDIA Management Library when done
    def shutdown():
        nvmlShutdown()
    
    # Return tic and toc functions, and the memory usage array
    return tic, toc, memory_usage, shutdown