# power_monitor.py - Windows Version (GPU + CPU Estimation)
# FIXED VERSION: Increased sampling rate from 0.5s to 0.1s for better accuracy
import csv
import subprocess
import time
import os
import threading
import re

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not installed. Run: pip install psutil")


class PowerMonitor:
    """
    Windows power monitor using nvidia-smi for GPU and psutil for CPU estimation.
    """

    def __init__(self, log_path: str):
        self.log_path = os.path.abspath(log_path)
        self.running = False
        self.proc = None
        self.thread = None
        self.current_task_id = "idle"
        self.start_time = None
        self.csv_file = None
        self.writer = None

        # CPU power estimation (5800X3D TDP = 105W)
        self.cpu_tdp = 105.0  # Watts

        # Check nvidia-smi availability
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                print(f"PowerMonitor initialized:")
                print(f"  GPU: {gpu_name} (nvidia-smi)")
                if PSUTIL_AVAILABLE:
                    print(f"  CPU: AMD Ryzen 7 5800X3D (estimated, TDP={self.cpu_tdp}W)")
                else:
                    print(f"  CPU: Not available (psutil not installed)")
                self.nvidia_smi_available = True
            else:
                print("ERROR: nvidia-smi command failed")
                self.nvidia_smi_available = False
        except Exception as e:
            print(f"ERROR: Could not run nvidia-smi: {e}")
            self.nvidia_smi_available = False

        # Open CSV file and write header
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self.csv_file = open(self.log_path, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(["task_id", "time_s", "cpu_w", "gpu_w", "total_w"])
            print(f"Power logging to: {self.log_path}")
        except Exception as e:
            print(f"ERROR: Could not open log file {self.log_path}: {e}")
            self.csv_file = None
            self.writer = None

    def _estimate_cpu_power(self) -> float:
        """Estimate CPU power based on utilization."""
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            # Get CPU usage percentage (non-blocking, quick sample)
            # This returns usage since the last call (or system boot on first call)
            cpu_percent = psutil.cpu_percent(interval=0.0)

            # Estimate power: base idle (~30W) + proportional load
            # Idle: ~30W, Full load: ~105W
            idle_power = 30.0
            max_power = self.cpu_tdp

            # Linear estimation
            estimated_power = idle_power + (cpu_percent / 100.0) * (max_power - idle_power)
            return estimated_power
        except Exception:
            return 0.0

    def _sample_loop(self):
        """Background thread that continuously queries nvidia-smi and estimates CPU."""
        # Start nvidia-smi in continuous mode
        cmd = [
            "nvidia-smi",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            "-l", "0.1"  # FIXED: Loop every 0.1 seconds (100ms) for better capture of short tasks
        ]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Read GPU power output line by line
            for line in iter(self.proc.stdout.readline, ''):
                if not self.running:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse GPU power (in watts)
                    gpu_w = float(line)

                    # Estimate CPU power
                    cpu_w = self._estimate_cpu_power()

                    # Calculate total
                    total_w = cpu_w + gpu_w

                    # Calculate elapsed time
                    elapsed = time.perf_counter() - self.start_time

                    # Write to CSV
                    if self.writer:
                        self.writer.writerow([
                            self.current_task_id,
                            round(elapsed, 2),
                            round(cpu_w, 2),
                            round(gpu_w, 2),
                            round(total_w, 2)
                        ])
                        self.csv_file.flush()

                except ValueError:
                    continue

        except Exception as e:
            if self.running:
                print(f"Error in power monitoring: {e}")
        finally:
            if self.proc:
                self.proc.stdout.close()

    def start(self, task_id: str):
        """Start monitoring power for a specific task."""
        if not self.nvidia_smi_available or not self.writer:
            return

        if self.running:
            print("WARNING: Monitor already running. Call stop() first.")
            return

        self.current_task_id = task_id
        self.start_time = time.perf_counter()

        # Call psutil.cpu_percent once before starting to establish a baseline
        # This prevents the first reading from being the average since system boot
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)

        self.running = True

        # Start background sampling thread
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        """Stop monitoring and return duration."""
        if not self.running:
            return 0.0

        duration = time.perf_counter() - self.start_time
        self.running = False

        # Terminate nvidia-smi process
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

        return duration

    def close(self):
        """Clean up resources."""
        if self.running:
            self.stop()

        if self.csv_file:
            self.csv_file.close()
            print(f"PowerMonitor closed. Log saved to: {self.log_path}")