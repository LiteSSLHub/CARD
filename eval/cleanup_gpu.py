#!/usr/bin/env python3
"""
GPU Cleanup Tool - For checking and cleaning stuck GPU processes
Use when nvidia-smi hangs or GPU is occupied by zombie processes
"""

import subprocess
import sys
import time


def check_nvidia_smi():
    """Check if nvidia-smi responds"""
    print("Checking nvidia-smi response...")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ nvidia-smi responds normally")
            return True
        else:
            print(f"✗ nvidia-smi returned error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ nvidia-smi timed out - GPU driver may be hung")
        return False
    except Exception as e:
        print(f"✗ Error running nvidia-smi: {str(e)}")
        return False


def list_gpu_processes():
    """List all processes using GPU"""
    print("\nCurrent GPU processes:")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(result.stdout)
            return True
    except Exception as e:
        print(f"Failed to get GPU processes: {str(e)}")
    return False


def kill_gpu_processes():
    """Kill all compute processes using GPU"""
    print("\nTerminating GPU processes...")
    try:
        # Get all GPU compute processes
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            pids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
            
            if not pids:
                print("No GPU processes found")
                return
            
            for pid in pids:
                try:
                    # Get process info
                    proc_result = subprocess.run(
                        ["ps", "-fp", str(pid)],
                        capture_output=True,
                        text=True
                    )
                    print(f"\nTerminating process PID={pid}:")
                    print(proc_result.stdout)
                    
                    # Kill process
                    subprocess.run(["kill", "-9", str(pid)], check=False)
                    print(f"  ✓ terminated")
                except Exception as e:
                    print(f"  ✗ termination failed: {str(e)}")
        
        time.sleep(2)
        print("\nCleanup complete")
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")


def kill_vllm_processes():
    """Kill vLLM-related processes specifically"""
    print("\nFinding and terminating vLLM-related processes...")
    try:
        # Find vLLM processes
        result = subprocess.run(
            ["pgrep", "-f", "vllm.*api_server"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                pid = pid.strip()
                if pid:
                    print(f"Terminating vLLM process PID={pid}")
                    subprocess.run(["kill", "-9", pid], check=False)
        else:
            print("No vLLM processes found")
            
        # Find VllmWorker processes
        result = subprocess.run(
            ["pgrep", "-f", "VllmWorker"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                pid = pid.strip()
                if pid:
                    print(f"Terminating VllmWorker process PID={pid}")
                    subprocess.run(["kill", "-9", pid], check=False)
        
        time.sleep(2)
        print("vLLM process cleanup complete")
        
    except Exception as e:
        print(f"Error cleaning vLLM processes: {str(e)}")


def main():
    print("=" * 60)
    print("GPU Cleanup Tool")
    print("=" * 60)
    
    # Check nvidia-smi
    if check_nvidia_smi():
        list_gpu_processes()
    
    print("\nOptions:")
    print("1. Kill all GPU compute processes")
    print("2. Kill vLLM-related processes")
    print("3. Perform full cleanup (GPU processes + vLLM processes)")
    print("4. Check only, do not cleanup")
    print("q. Quit")
    
    choice = input("\nPlease choose [1/2/3/4/q]: ").strip().lower()
    
    if choice == '1':
        kill_gpu_processes()
    elif choice == '2':
        kill_vllm_processes()
    elif choice == '3':
        kill_gpu_processes()
        kill_vllm_processes()
        print("\nRechecking GPU status...")
        check_nvidia_smi()
    elif choice == '4':
        check_nvidia_smi()
    else:
        print("Quit")
        return
    
    print("\nDone. If GPU is still unresponsive, machine restart may be needed.")


if __name__ == "__main__":
    main()
