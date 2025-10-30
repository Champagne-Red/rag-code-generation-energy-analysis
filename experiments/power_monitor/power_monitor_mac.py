# power_monitor.py
import csv
import subprocess
import time
import os
import re
import threading


class PowerMonitor:
    """
    A class to start, stop, and log powermetrics data in a background thread.
    """

    def __init__(self, log_path: str):
        self.log_path = os.path.abspath(log_path)
        self.proc = None
        self.thread = None
        self.running = False
        self.current_task_id = "idle"
        self.start_time = None
        self.csv_file = None
        self.writer = None

        # Regex to find the power lines (e.g., "CPU Power: 123 mW")
        self.cpu_regex = re.compile(r"CPU Power:\s+([\d\.]+)\s+mW")
        self.gpu_regex = re.compile(r"GPU Power:\s+([\d\.]+)\s+mW")
        # This regex is our reliable trigger
        self.trigger_regex = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):")

        # Open file and write header
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self.csv_file = open(self.log_path, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(["task_id", "time_s", "cpu_w", "gpu_w"])
            print(f"PowerMonitor initialized. Logging to {self.log_path}")
        except Exception as e:
            print(f"FATAL: Could not open log file {self.log_path}. Error: {e}")
            print("Please check permissions.")

    def _read_loop(self):
        """Internal method to run in a thread, reading powermetrics output."""
        current_power = {"cpu": None, "gpu": None}

        try:
            for line in iter(self.proc.stdout.readline, ''):
                if not self.running:
                    break

                cpu_match = self.cpu_regex.search(line)
                gpu_match = self.gpu_regex.search(line)
                trigger_match = self.trigger_regex.search(line)

                if cpu_match and current_power["cpu"] is None:
                    current_power["cpu"] = float(cpu_match.group(1)) / 1000
                elif gpu_match and current_power["gpu"] is None:
                    current_power["gpu"] = float(gpu_match.group(1)) / 1000

                if trigger_match:
                    if all(v is not None for v in current_power.values()):
                        elapsed = time.perf_counter() - self.start_time
                        cpu = current_power["cpu"]
                        gpu = current_power["gpu"]

                        self.writer.writerow([
                            self.current_task_id,
                            round(elapsed, 2),
                            cpu,
                            gpu
                        ])

                    # Always reset for the next block
                    current_power = {"cpu": None, "gpu": None}

        except Exception as e:
            if self.running:  # Don't print error if we just stopped
                print(f"Error in powermetrics read loop: {e}")
        finally:
            self.proc.stdout.close()

    def start(self, task_id: str):
        """Starts monitoring powermetrics for a specific task."""
        if self.running:
            print("Warning: Monitor already running. Please call stop() first.")
            return

        self.current_task_id = task_id
        self.start_time = time.perf_counter()
        self.running = True

        powermetrics_cmd = [
            "sudo", "powermetrics",
            "-i", "500",  # 500ms interval
            "-a", "1"  # Print power average every 1 sample
        ]

        try:
            self.proc = subprocess.Popen(
                powermetrics_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='ignore'
            )

            self.thread = threading.Thread(target=self._read_loop)
            self.thread.start()
            # print(f"Monitoring started for {task_id}...")

        except Exception as e:
            print(f"Error starting powermetrics: {e}")
            self.running = False

    def stop(self) -> float:
        """Stops monitoring and returns the duration."""
        if not self.running:
            return 0.0

        duration = time.perf_counter() - self.start_time
        # print(f"Monitoring stopped for {self.current_task_id}. Duration: {duration:.3f}s")

        self.running = False
        if self.proc:
            self.proc.terminate()

        if self.thread:
            self.thread.join(timeout=1.0)

        try:
            if self.proc:
                self.proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            if self.proc:
                self.proc.kill()
                # print("Forcibly killed powermetrics.")

        self.proc = None
        self.thread = None

        return duration

    def close(self):
        """Closes the log file."""
        if self.csv_file:
            self.csv_file.close()
            print(f"PowerMonitor closed.")