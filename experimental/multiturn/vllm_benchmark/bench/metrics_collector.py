"""
Metrics collector for vLLM server during benchmarks.
Polls /metrics endpoint and generates visualizations.
"""

import asyncio
import csv
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import matplotlib.pyplot as plt


@dataclass
class GpuTransferSnapshot:
    timestamp: float
    gpu_id: int = 0
    tx_pci: float = 0.0  # PCIe TX (MB/s)
    rx_pci: float = 0.0  # PCIe RX (MB/s)


class GpuTransferCollector:
    """DEPRECATED: Collects GPU transfer stats using nvidia-smi dmon.

    Replaced by vLLM's native kv_offload metrics (vllm:kv_offload_total_bytes_total,
    vllm:kv_offload_total_time_total) which are more precise and don't require
    spawning a subprocess.
    """

    def __init__(self, gpu_id: int = 0, poll_interval: int = 1):
        self.gpu_id = gpu_id
        self.poll_interval = poll_interval
        self.snapshots: list[GpuTransferSnapshot] = []
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def _parse_line(self, line: str) -> GpuTransferSnapshot | None:
        """Parse a line of nvidia-smi dmon CSV output.

        Format: gpu, rxpci, txpci (values in MB/s)
        Example: 0, 406, 32013
        """
        line = line.strip()
        if not line or line.startswith('#'):  # Skip header/comments
            return None

        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            return None

        try:
            return GpuTransferSnapshot(
                timestamp=time.time(),
                gpu_id=int(parts[0]),
                rx_pci=float(parts[1]) if parts[1] != '-' else 0.0,
                tx_pci=float(parts[2]) if parts[2] != '-' else 0.0,
            )
        except (ValueError, IndexError):
            return None

    def _reader_thread(self) -> None:
        """Background thread to read nvidia-smi output."""
        if self._process is None:
            return

        for line in iter(self._process.stdout.readline, ''):
            if not self._running:
                break
            snapshot = self._parse_line(line)
            if snapshot and snapshot.gpu_id == self.gpu_id:
                self.snapshots.append(snapshot)

    def start(self) -> None:
        """Start collecting GPU transfer stats."""
        if self._running:
            return

        self._running = True
        self.snapshots = []

        try:
            self._process = subprocess.Popen(
                [
                    'nvidia-smi', 'dmon',
                    '-i', str(self.gpu_id),
                    '-s', 't',
                    '-d', str(self.poll_interval),
                    '--format', 'csv',
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._thread = threading.Thread(target=self._reader_thread, daemon=True)
            self._thread.start()
        except FileNotFoundError:
            print("nvidia-smi not found, GPU transfer monitoring disabled")
            self._running = False

    def stop(self) -> None:
        """Stop collecting GPU transfer stats."""
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None


@dataclass
class MetricsSnapshot:
    timestamp: float
    kv_cache_usage: float = 0.0
    cpu_kv_cache_usage: float = 0.0
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_queries: int = 0
    cpu_prefix_cache_hits: int = 0
    cpu_prefix_cache_queries: int = 0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    num_preemptions: int = 0
    request_success: int = 0
    # KV offload transfer metrics (cumulative)
    kv_offload_bytes_gpu_to_cpu: float = 0.0
    kv_offload_bytes_cpu_to_gpu: float = 0.0
    kv_offload_time_gpu_to_cpu: float = 0.0
    kv_offload_time_cpu_to_gpu: float = 0.0
    # Prompt tokens by source (cumulative)
    prompt_tokens_local_compute: int = 0
    prompt_tokens_local_cache_hit: int = 0
    prompt_tokens_external_kv_transfer: int = 0
    # Prefill KV computed tokens (cumulative sum from histogram)
    prefill_kv_computed_tokens_sum: int = 0
    prefill_kv_computed_tokens_count: int = 0


@dataclass
class MetricsCollector:
    base_url: str
    poll_interval: float = 1.0
    snapshots: list[MetricsSnapshot] = field(default_factory=list)
    _running: bool = False
    _task: asyncio.Task | None = None
    gpu_transfer_collector: GpuTransferCollector | None = None
    gpu_id: int = 0

    def _parse_metrics(self, text: str) -> MetricsSnapshot:
        """Parse Prometheus metrics text format."""
        snapshot = MetricsSnapshot(timestamp=time.time())

        # Helper to extract gauge/counter value
        def get_value(pattern: str, default: float = 0.0) -> float:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
            return default

        # KV cache usage (0-1 scale)
        snapshot.kv_cache_usage = get_value(
            r'vllm:gpu_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)'
        )
        # Fallback to old metric name if new one not found
        if snapshot.kv_cache_usage == 0.0:
            snapshot.kv_cache_usage = get_value(
                r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)'
            )

        # CPU/offloaded KV cache usage
        snapshot.cpu_kv_cache_usage = get_value(
            r'vllm:cpu_cache_usage_perc\{[^}]*\}\s+([\d.e+-]+)'
        )

        # Running/waiting requests
        snapshot.num_requests_running = int(get_value(
            r'vllm:num_requests_running\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.num_requests_waiting = int(get_value(
            r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Prefix cache (cumulative counters) - GPU
        snapshot.prefix_cache_hits = int(get_value(
            r'vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.prefix_cache_queries = int(get_value(
            r'vllm:prefix_cache_queries_total\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Prefix cache - external/offloaded (KV connector cross-instance cache)
        snapshot.cpu_prefix_cache_hits = int(get_value(
            r'vllm:external_prefix_cache_hits_total\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.cpu_prefix_cache_queries = int(get_value(
            r'vllm:external_prefix_cache_queries_total\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Token counters
        snapshot.prompt_tokens = int(get_value(
            r'vllm:prompt_tokens_total\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.generation_tokens = int(get_value(
            r'vllm:generation_tokens_total\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Preemptions
        snapshot.num_preemptions = int(get_value(
            r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.e+-]+)'
        ))

        # Request success (sum all finish reasons)
        for match in re.finditer(
            r'vllm:request_success_total\{[^}]*finished_reason="[^"]*"[^}]*\}\s+([\d.e+-]+)',
            text
        ):
            snapshot.request_success += int(float(match.group(1)))

        # KV offload bytes transferred (cumulative counters by direction)
        snapshot.kv_offload_bytes_gpu_to_cpu = get_value(
            r'vllm:kv_offload_total_bytes_total\{[^}]*transfer_type="GPU_to_CPU"[^}]*\}\s+([\d.e+-]+)'
        )
        snapshot.kv_offload_bytes_cpu_to_gpu = get_value(
            r'vllm:kv_offload_total_bytes_total\{[^}]*transfer_type="CPU_to_GPU"[^}]*\}\s+([\d.e+-]+)'
        )

        # KV offload time (cumulative, seconds)
        snapshot.kv_offload_time_gpu_to_cpu = get_value(
            r'vllm:kv_offload_total_time_total\{[^}]*transfer_type="GPU_to_CPU"[^}]*\}\s+([\d.e+-]+)'
        )
        snapshot.kv_offload_time_cpu_to_gpu = get_value(
            r'vllm:kv_offload_total_time_total\{[^}]*transfer_type="CPU_to_GPU"[^}]*\}\s+([\d.e+-]+)'
        )

        # Prompt tokens by source (cumulative)
        snapshot.prompt_tokens_local_compute = int(get_value(
            r'vllm:prompt_tokens_by_source_total\{[^}]*source="local_compute"[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.prompt_tokens_local_cache_hit = int(get_value(
            r'vllm:prompt_tokens_by_source_total\{[^}]*source="local_cache_hit"[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.prompt_tokens_external_kv_transfer = int(get_value(
            r'vllm:prompt_tokens_by_source_total\{[^}]*source="external_kv_transfer"[^}]*\}\s+([\d.e+-]+)'
        ))

        # Prefill KV computed tokens (histogram sum and count)
        snapshot.prefill_kv_computed_tokens_sum = int(get_value(
            r'vllm:request_prefill_kv_computed_tokens_sum\{[^}]*\}\s+([\d.e+-]+)'
        ))
        snapshot.prefill_kv_computed_tokens_count = int(get_value(
            r'vllm:request_prefill_kv_computed_tokens_count\{[^}]*\}\s+([\d.e+-]+)'
        ))

        return snapshot

    async def _poll_loop(self) -> None:
        """Background polling loop."""
        metrics_url = f"{self.base_url}/metrics"
        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            snapshot = self._parse_metrics(text)
                            self.snapshots.append(snapshot)
                except Exception as e:
                    print(f"Metrics poll error: {e}")

                await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        """Start background metrics collection."""
        if self._running:
            return
        self._running = True
        self.snapshots = []
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop metrics collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def generate_plots(
        self,
        output_prefix: str = "metrics",
        client_metrics: list | None = None,
    ) -> None:
        """Generate visualization plots from collected metrics.

        Args:
            output_prefix: Prefix for output file names
            client_metrics: Optional list of RequestStats from benchmark clients
        """
        if len(self.snapshots) < 2:
            print("Not enough data points for plots")
            return

        # Convert to relative time (seconds from start)
        start_time = self.snapshots[0].timestamp
        times = [(s.timestamp - start_time) for s in self.snapshots]

        # Create figure with subplots
        num_rows = 6 if client_metrics else 4
        fig, axes = plt.subplots(num_rows, 2, figsize=(14, 4 * num_rows))
        fig.suptitle("vLLM Server Metrics During Benchmark", fontsize=14)

        # 1. KV Cache Usage vs Time
        ax = axes[0, 0]
        kv_usage = [s.kv_cache_usage * 100 for s in self.snapshots]
        ax.plot(times, kv_usage, 'b-', label='GPU', linewidth=1.5)
        # Add external cache if available
        cpu_kv_usage = [s.cpu_kv_cache_usage * 100 for s in self.snapshots]
        if any(v > 0 for v in cpu_kv_usage):
            ax.plot(times, cpu_kv_usage, 'r--', label='External', linewidth=1.5)
            ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("KV Cache Usage (%)")
        ax.set_title("KV Cache Utilization Over Time")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # 2. Running & Waiting Requests vs Time
        ax = axes[0, 1]
        running = [s.num_requests_running for s in self.snapshots]
        waiting = [s.num_requests_waiting for s in self.snapshots]
        ax.plot(times, running, 'g-', label='Running', linewidth=1.5)
        ax.plot(times, waiting, 'r-', label='Waiting', linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Requests")
        ax.set_title("Request Queue Depth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Cache Hit Rate vs Time (computed from deltas between polling intervals)
        ax = axes[1, 0]
        gpu_hit_rates = []
        ext_hit_rates = []
        combined_hit_rates = []
        has_ext_cache = any(s.cpu_prefix_cache_queries > 0 for s in self.snapshots)
        for i in range(1, len(self.snapshots)):
            # GPU (HBM) cache hit rate for this interval
            gpu_delta_hits = self.snapshots[i].prefix_cache_hits - self.snapshots[i-1].prefix_cache_hits
            gpu_delta_queries = self.snapshots[i].prefix_cache_queries - self.snapshots[i-1].prefix_cache_queries
            if gpu_delta_queries > 0:
                gpu_hit_rates.append(100.0 * gpu_delta_hits / gpu_delta_queries)
            else:
                gpu_hit_rates.append(gpu_hit_rates[-1] if gpu_hit_rates else 0)

            # External cache hit rate for this interval
            if has_ext_cache:
                ext_delta_hits = self.snapshots[i].cpu_prefix_cache_hits - self.snapshots[i-1].cpu_prefix_cache_hits
                ext_delta_queries = self.snapshots[i].cpu_prefix_cache_queries - self.snapshots[i-1].cpu_prefix_cache_queries
                if ext_delta_queries > 0:
                    ext_hit_rates.append(100.0 * ext_delta_hits / ext_delta_queries)
                else:
                    ext_hit_rates.append(ext_hit_rates[-1] if ext_hit_rates else 0)

                # Combined hit rate: (gpu_hits + ext_hits) / (gpu_queries + ext_queries)
                total_hits = gpu_delta_hits + ext_delta_hits
                total_queries = gpu_delta_queries + ext_delta_queries
                if total_queries > 0:
                    combined_hit_rates.append(100.0 * total_hits / total_queries)
                else:
                    combined_hit_rates.append(combined_hit_rates[-1] if combined_hit_rates else 0)

        # Rolling window size
        window = min(50, len(gpu_hit_rates) // 10) if len(gpu_hit_rates) > 10 else 1

        # Scatter plot for GPU (HBM) cache hit rate
        ax.scatter(times[1:], gpu_hit_rates, alpha=0.3, s=5, c='purple', label='GPU (HBM)')
        if window > 1:
            rolling_gpu = [
                sum(gpu_hit_rates[max(0, i - window):i + 1]) / len(gpu_hit_rates[max(0, i - window):i + 1])
                for i in range(len(gpu_hit_rates))
            ]
            ax.plot(times[1:], rolling_gpu, 'purple', linewidth=1.5, label=f'GPU avg (n={window})')

        # External cache scatter + rolling (if available)
        if has_ext_cache and ext_hit_rates:
            ax.scatter(times[1:], ext_hit_rates, alpha=0.3, s=5, c='orange', label='External')
            if window > 1:
                rolling_ext = [
                    sum(ext_hit_rates[max(0, i - window):i + 1]) / len(ext_hit_rates[max(0, i - window):i + 1])
                    for i in range(len(ext_hit_rates))
                ]
                ax.plot(times[1:], rolling_ext, 'orange', linewidth=1.5, label=f'External avg (n={window})')

            # Combined/total hit rate (only if external exists)
            ax.scatter(times[1:], combined_hit_rates, alpha=0.2, s=3, c='green', label='Combined')
            if window > 1:
                rolling_combined = [
                    sum(combined_hit_rates[max(0, i - window):i + 1]) / len(combined_hit_rates[max(0, i - window):i + 1])
                    for i in range(len(combined_hit_rates))
                ]
                ax.plot(times[1:], rolling_combined, 'green', linewidth=2, label=f'Combined avg (n={window})')

        ax.legend(loc='best', fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hit Rate (%)")
        ax.set_title("Prefix Cache Hit Rate Per Interval (tokens hit / tokens queried)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # 4. Throughput vs Time (tokens/sec) with rolling average
        ax = axes[1, 1]
        throughputs = []
        for i in range(1, len(self.snapshots)):
            delta_gen = self.snapshots[i].generation_tokens - self.snapshots[i-1].generation_tokens
            delta_time = self.snapshots[i].timestamp - self.snapshots[i-1].timestamp
            if delta_time > 0:
                throughputs.append(delta_gen / delta_time)
            else:
                throughputs.append(0)
        ax.scatter(times[1:], throughputs, alpha=0.15, s=3, c='orange')
        window = min(30, len(throughputs) // 10) if len(throughputs) > 10 else 1
        if window > 1:
            rolling_tp = [
                sum(throughputs[max(0, i - window):i + 1]) / len(throughputs[max(0, i - window):i + 1])
                for i in range(len(throughputs))
            ]
            ax.plot(times[1:], rolling_tp, 'orange', linewidth=1.5, label=f'Rolling avg (n={window})')
            ax.legend(fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Generation Throughput")
        ax.grid(True, alpha=0.3)

        # 5. KV Offload Transfer Rate (from vLLM metrics)
        ax = axes[2, 0]
        gpu_to_cpu_rates = []
        cpu_to_gpu_rates = []
        for i in range(1, len(self.snapshots)):
            dt = self.snapshots[i].timestamp - self.snapshots[i-1].timestamp
            if dt > 0:
                delta_g2c = self.snapshots[i].kv_offload_bytes_gpu_to_cpu - self.snapshots[i-1].kv_offload_bytes_gpu_to_cpu
                delta_c2g = self.snapshots[i].kv_offload_bytes_cpu_to_gpu - self.snapshots[i-1].kv_offload_bytes_cpu_to_gpu
                gpu_to_cpu_rates.append(delta_g2c / dt / 1e6)  # MB/s
                cpu_to_gpu_rates.append(delta_c2g / dt / 1e6)  # MB/s
            else:
                gpu_to_cpu_rates.append(0)
                cpu_to_gpu_rates.append(0)
        if any(r > 0 for r in gpu_to_cpu_rates) or any(r > 0 for r in cpu_to_gpu_rates):
            ax.scatter(times[1:], gpu_to_cpu_rates, alpha=0.15, s=3, c='blue')
            ax.scatter(times[1:], cpu_to_gpu_rates, alpha=0.15, s=3, c='red')
            xfer_window = min(30, len(gpu_to_cpu_rates) // 10) if len(gpu_to_cpu_rates) > 10 else 1
            if xfer_window > 1:
                rolling_g2c = [
                    sum(gpu_to_cpu_rates[max(0, i - xfer_window):i + 1]) / len(gpu_to_cpu_rates[max(0, i - xfer_window):i + 1])
                    for i in range(len(gpu_to_cpu_rates))
                ]
                rolling_c2g = [
                    sum(cpu_to_gpu_rates[max(0, i - xfer_window):i + 1]) / len(cpu_to_gpu_rates[max(0, i - xfer_window):i + 1])
                    for i in range(len(cpu_to_gpu_rates))
                ]
                ax.plot(times[1:], rolling_g2c, 'b-', linewidth=1.5, label=f'GPU→CPU (avg n={xfer_window})')
                ax.plot(times[1:], rolling_c2g, 'r-', linewidth=1.5, label=f'CPU→GPU (avg n={xfer_window})')
            else:
                ax.plot(times[1:], gpu_to_cpu_rates, 'b-', linewidth=1, alpha=0.8, label='GPU→CPU')
                ax.plot(times[1:], cpu_to_gpu_rates, 'r-', linewidth=1, alpha=0.8, label='CPU→GPU')
            ax.legend(fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Transfer Rate (MB/s)")
        ax.set_title("KV Offload Transfer Rate")
        ax.grid(True, alpha=0.3)

        # 6. Prompt Token Sources Over Time (cumulative percentage)
        ax = axes[2, 1]
        initial = self.snapshots[0]
        cum_compute_pct = []
        cum_cache_pct = []
        cum_ext_pct = []
        for s in self.snapshots:
            c = s.prompt_tokens_local_compute - initial.prompt_tokens_local_compute
            h = s.prompt_tokens_local_cache_hit - initial.prompt_tokens_local_cache_hit
            e = s.prompt_tokens_external_kv_transfer - initial.prompt_tokens_external_kv_transfer
            total = c + h + e
            if total > 0:
                cum_compute_pct.append(100.0 * c / total)
                cum_cache_pct.append(100.0 * h / total)
                cum_ext_pct.append(100.0 * e / total)
            else:
                cum_compute_pct.append(0)
                cum_cache_pct.append(0)
                cum_ext_pct.append(0)
        if any(v > 0 for v in cum_compute_pct):
            ax.stackplot(times, cum_compute_pct, cum_cache_pct, cum_ext_pct,
                        labels=['Local Compute', 'Local Cache Hit', 'External KV Transfer'],
                        colors=['coral', 'steelblue', 'mediumseagreen'], alpha=0.8)
            ax.legend(fontsize=8, loc='lower left')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("% of Prefill Tokens")
        ax.set_title("Cumulative Prefill Token Source Breakdown")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # 7. Cumulative KV Offload Transfers
        initial = self.snapshots[0]
        # GPU → CPU cumulative
        ax = axes[3, 0]
        cum_g2c = [(s.kv_offload_bytes_gpu_to_cpu - initial.kv_offload_bytes_gpu_to_cpu) / 1e9
                    for s in self.snapshots]
        if any(v > 0 for v in cum_g2c):
            ax.plot(times, cum_g2c, 'b-', linewidth=1.5)
            ax.fill_between(times, cum_g2c, alpha=0.2, color='blue')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Transfer (GB)")
        ax.set_title("KV Offload: GPU → CPU (Cumulative)")
        ax.grid(True, alpha=0.3)

        # CPU → GPU cumulative
        ax = axes[3, 1]
        cum_c2g = [(s.kv_offload_bytes_cpu_to_gpu - initial.kv_offload_bytes_cpu_to_gpu) / 1e9
                    for s in self.snapshots]
        if any(v > 0 for v in cum_c2g):
            ax.plot(times, cum_c2g, 'r-', linewidth=1.5)
            ax.fill_between(times, cum_c2g, alpha=0.2, color='red')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Transfer (GB)")
        ax.set_title("KV Offload: CPU → GPU (Cumulative)")
        ax.grid(True, alpha=0.3)

        # 8 & 9. Client metrics plots (TTFT and Latency vs Time)
        if client_metrics and len(client_metrics) > 0:
            # Sort by start time
            sorted_metrics = sorted(client_metrics, key=lambda x: x.start_time_ms)
            # Align client times to server start_time so x-axis matches server plots
            server_start_ms = start_time * 1000.0
            request_times = [(m.start_time_ms - server_start_ms) / 1000.0 for m in sorted_metrics]
            ttfts = [m.ttft_ms for m in sorted_metrics]
            latencies = [m.latency_ms for m in sorted_metrics]

            # 8. TTFT vs Time
            ax = axes[4, 0]
            ax.scatter(request_times, ttfts, alpha=0.3, s=5, c='blue')
            # Add rolling average
            window = min(50, len(ttfts) // 10) if len(ttfts) > 10 else 1
            if window > 1:
                rolling_ttft = [
                    sum(ttfts[max(0, i - window):i + 1]) / len(ttfts[max(0, i - window):i + 1])
                    for i in range(len(ttfts))
                ]
                ax.plot(request_times, rolling_ttft, 'r-', linewidth=1.5, label=f'Rolling avg (n={window})')
                ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("TTFT (ms)")
            ax.set_title("Time to First Token vs Time")
            ax.grid(True, alpha=0.3)

            # 9. Latency vs Time
            ax = axes[4, 1]
            ax.scatter(request_times, latencies, alpha=0.3, s=5, c='green')
            # Add rolling average
            if window > 1:
                rolling_latency = [
                    sum(latencies[max(0, i - window):i + 1]) / len(latencies[max(0, i - window):i + 1])
                    for i in range(len(latencies))
                ]
                ax.plot(request_times, rolling_latency, 'r-', linewidth=1.5, label=f'Rolling avg (n={window})')
                ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Request Latency vs Time")
            ax.grid(True, alpha=0.3)

            # 10. Interactivity (1/TPOT = tokens/sec) vs Time
            ax = axes[5, 0]
            # Filter out zero TPOT values to avoid division by zero
            tpots = [m.tpot_ms for m in sorted_metrics]
            interactivity = [1000.0 / t if t > 0 else 0 for t in tpots]  # Convert to tokens/sec
            ax.scatter(request_times, interactivity, alpha=0.3, s=5, c='purple')
            # Add rolling average
            if window > 1:
                rolling_inter = [
                    sum(interactivity[max(0, i - window):i + 1]) / len(interactivity[max(0, i - window):i + 1])
                    for i in range(len(interactivity))
                ]
                ax.plot(request_times, rolling_inter, 'r-', linewidth=1.5, label=f'Rolling avg (n={window})')
                ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Interactivity (tokens/sec)")
            ax.set_title("Decode Speed (1/TPOT) vs Time")
            ax.grid(True, alpha=0.3)

            # 11. Preemptions over time
            ax = axes[5, 1]
            preemption_rates = []
            for i in range(1, len(self.snapshots)):
                dt = self.snapshots[i].timestamp - self.snapshots[i-1].timestamp
                delta = self.snapshots[i].num_preemptions - self.snapshots[i-1].num_preemptions
                preemption_rates.append(delta / dt if dt > 0 else 0)
            if any(r > 0 for r in preemption_rates):
                ax.scatter(times[1:], preemption_rates, alpha=0.15, s=3, c='red')
                preempt_window = min(30, len(preemption_rates) // 10) if len(preemption_rates) > 10 else 1
                if preempt_window > 1:
                    rolling_preempt = [
                        sum(preemption_rates[max(0, i - preempt_window):i + 1]) / len(preemption_rates[max(0, i - preempt_window):i + 1])
                        for i in range(len(preemption_rates))
                    ]
                    ax.plot(times[1:], rolling_preempt, 'r-', linewidth=1.5, label=f'Rolling avg (n={preempt_window})')
                # Cumulative on secondary axis
                ax2 = ax.twinx()
                cumulative = [self.snapshots[i].num_preemptions - self.snapshots[0].num_preemptions
                              for i in range(1, len(self.snapshots))]
                ax2.plot(times[1:], cumulative, 'b--', linewidth=1, alpha=0.5, label='Cumulative')
                ax2.set_ylabel("Cumulative Preemptions", color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Preemptions/sec", color='red')
            ax.tick_params(axis='y', labelcolor='red')
            ax.set_title("Preemptions Over Time")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_plots.png", dpi=150)
        print(f"Saved plots to {output_prefix}_plots.png")
        plt.close()

        # Also generate a summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print summary statistics."""
        if len(self.snapshots) < 2:
            return

        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        total_gen_tokens = self.snapshots[-1].generation_tokens - self.snapshots[0].generation_tokens
        total_prompt_tokens = self.snapshots[-1].prompt_tokens - self.snapshots[0].prompt_tokens

        final = self.snapshots[-1]
        initial = self.snapshots[0]

        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        print(f"Duration: {duration:.1f}s")
        print(f"Total prompt tokens: {total_prompt_tokens:,}")
        print(f"Total generation tokens: {total_gen_tokens:,}")
        print(f"Avg generation throughput: {total_gen_tokens/duration:.1f} tok/s")
        print(f"Peak KV cache usage: {max(s.kv_cache_usage for s in self.snapshots)*100:.1f}%")
        print(f"Peak running requests: {max(s.num_requests_running for s in self.snapshots)}")
        print(f"Peak waiting requests: {max(s.num_requests_waiting for s in self.snapshots)}")
        print(f"Total preemptions: {final.num_preemptions - initial.num_preemptions}")

        if final.prefix_cache_queries > initial.prefix_cache_queries:
            delta_hits = final.prefix_cache_hits - initial.prefix_cache_hits
            delta_queries = final.prefix_cache_queries - initial.prefix_cache_queries
            hit_rate = 100.0 * delta_hits / delta_queries
            print(f"Overall GPU cache hit rate: {hit_rate:.1f}%")
            print(f"  - Cache hits: {delta_hits:,} tokens")
            print(f"  - Cache queries: {delta_queries:,} tokens")

        # External/offloaded cache stats if available
        if final.cpu_prefix_cache_queries > initial.cpu_prefix_cache_queries:
            cpu_delta_hits = final.cpu_prefix_cache_hits - initial.cpu_prefix_cache_hits
            cpu_delta_queries = final.cpu_prefix_cache_queries - initial.cpu_prefix_cache_queries
            cpu_hit_rate = 100.0 * cpu_delta_hits / cpu_delta_queries
            print(f"Overall external cache hit rate: {cpu_hit_rate:.1f}%")
            print(f"  - Cache hits: {cpu_delta_hits:,} tokens")
            print(f"  - Cache queries: {cpu_delta_queries:,} tokens")

        # Prompt tokens by source
        total_compute = final.prompt_tokens_local_compute - initial.prompt_tokens_local_compute
        total_cache_hit = final.prompt_tokens_local_cache_hit - initial.prompt_tokens_local_cache_hit
        total_ext = final.prompt_tokens_external_kv_transfer - initial.prompt_tokens_external_kv_transfer
        total_by_source = total_compute + total_cache_hit + total_ext
        if total_by_source > 0:
            print(f"Prompt token sources:")
            print(f"  - Local compute:      {total_compute:>12,} ({100*total_compute/total_by_source:.1f}%)")
            print(f"  - Local cache hit:    {total_cache_hit:>12,} ({100*total_cache_hit/total_by_source:.1f}%)")
            print(f"  - External KV xfer:   {total_ext:>12,} ({100*total_ext/total_by_source:.1f}%)")

        # KV offload transfer stats
        g2c_bytes = final.kv_offload_bytes_gpu_to_cpu - initial.kv_offload_bytes_gpu_to_cpu
        c2g_bytes = final.kv_offload_bytes_cpu_to_gpu - initial.kv_offload_bytes_cpu_to_gpu
        g2c_time = final.kv_offload_time_gpu_to_cpu - initial.kv_offload_time_gpu_to_cpu
        c2g_time = final.kv_offload_time_cpu_to_gpu - initial.kv_offload_time_cpu_to_gpu
        if g2c_bytes > 0 or c2g_bytes > 0:
            print(f"KV offload transfers:")
            print(f"  GPU→CPU: {g2c_bytes/1e9:.2f} GB in {g2c_time:.2f}s ({g2c_bytes/g2c_time/1e9:.1f} GB/s)" if g2c_time > 0 else f"  GPU→CPU: {g2c_bytes/1e9:.2f} GB")
            print(f"  CPU→GPU: {c2g_bytes/1e9:.2f} GB in {c2g_time:.2f}s ({c2g_bytes/c2g_time/1e9:.1f} GB/s)" if c2g_time > 0 else f"  CPU→GPU: {c2g_bytes/1e9:.2f} GB")

        # Prefill KV computed tokens
        delta_kv_sum = final.prefill_kv_computed_tokens_sum - initial.prefill_kv_computed_tokens_sum
        delta_kv_count = final.prefill_kv_computed_tokens_count - initial.prefill_kv_computed_tokens_count
        if delta_kv_count > 0:
            print(f"Prefill KV computed tokens (excluding cached):")
            print(f"  Total: {delta_kv_sum:,} tokens across {delta_kv_count:,} requests")
            print(f"  Avg per request: {delta_kv_sum/delta_kv_count:.0f} tokens")

        print("="*60 + "\n")

    def export_csv(
        self,
        output_prefix: str = "metrics",
        client_metrics: list | None = None,
    ) -> None:
        """Export all time series data to CSV files.

        Args:
            output_prefix: Prefix for output file names
            client_metrics: Optional list of RequestStats from benchmark clients

        Generates:
            - {output_prefix}_server_metrics.csv: vLLM server metrics over time
            - {output_prefix}_gpu_transfer.csv: GPU PCIe transfer stats
            - {output_prefix}_client_metrics.csv: Per-request client metrics (if provided)
        """
        output_dir = Path(output_prefix).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export server metrics (from /metrics endpoint)
        if self.snapshots:
            server_csv = f"{output_prefix}_server_metrics.csv"
            start_time = self.snapshots[0].timestamp

            with open(server_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow([
                    'timestamp_sec',
                    'relative_time_sec',
                    'kv_cache_usage_pct',
                    'cpu_kv_cache_usage_pct',
                    'num_requests_running',
                    'num_requests_waiting',
                    'prefix_cache_hits',
                    'prefix_cache_queries',
                    'cpu_prefix_cache_hits',
                    'cpu_prefix_cache_queries',
                    'prompt_tokens_total',
                    'generation_tokens_total',
                    'num_preemptions_total',
                    'request_success_total',
                    # KV offload metrics
                    'kv_offload_bytes_gpu_to_cpu',
                    'kv_offload_bytes_cpu_to_gpu',
                    'kv_offload_time_gpu_to_cpu',
                    'kv_offload_time_cpu_to_gpu',
                    # Prompt tokens by source
                    'prompt_tokens_local_compute',
                    'prompt_tokens_local_cache_hit',
                    'prompt_tokens_external_kv_transfer',
                    # Prefill KV computed
                    'prefill_kv_computed_tokens_sum',
                    'prefill_kv_computed_tokens_count',
                    # Computed per-interval metrics
                    'interval_cache_hit_rate_pct',
                    'interval_throughput_tok_per_sec',
                ])

                for i, s in enumerate(self.snapshots):
                    relative_time = s.timestamp - start_time

                    # Compute per-interval metrics
                    cache_hit_rate = 0.0
                    throughput = 0.0
                    if i > 0:
                        prev = self.snapshots[i - 1]
                        delta_hits = s.prefix_cache_hits - prev.prefix_cache_hits
                        delta_queries = s.prefix_cache_queries - prev.prefix_cache_queries
                        if delta_queries > 0:
                            cache_hit_rate = 100.0 * delta_hits / delta_queries

                        delta_gen = s.generation_tokens - prev.generation_tokens
                        delta_time = s.timestamp - prev.timestamp
                        if delta_time > 0:
                            throughput = delta_gen / delta_time

                    writer.writerow([
                        f"{s.timestamp:.3f}",
                        f"{relative_time:.3f}",
                        f"{s.kv_cache_usage * 100:.2f}",
                        f"{s.cpu_kv_cache_usage * 100:.2f}",
                        s.num_requests_running,
                        s.num_requests_waiting,
                        s.prefix_cache_hits,
                        s.prefix_cache_queries,
                        s.cpu_prefix_cache_hits,
                        s.cpu_prefix_cache_queries,
                        s.prompt_tokens,
                        s.generation_tokens,
                        s.num_preemptions,
                        s.request_success,
                        f"{s.kv_offload_bytes_gpu_to_cpu:.0f}",
                        f"{s.kv_offload_bytes_cpu_to_gpu:.0f}",
                        f"{s.kv_offload_time_gpu_to_cpu:.6f}",
                        f"{s.kv_offload_time_cpu_to_gpu:.6f}",
                        s.prompt_tokens_local_compute,
                        s.prompt_tokens_local_cache_hit,
                        s.prompt_tokens_external_kv_transfer,
                        s.prefill_kv_computed_tokens_sum,
                        s.prefill_kv_computed_tokens_count,
                        f"{cache_hit_rate:.2f}",
                        f"{throughput:.2f}",
                    ])

            print(f"Exported server metrics to {server_csv}")

        # 2. Export GPU transfer stats (DEPRECATED - kept for backward compat)
        if self.gpu_transfer_collector and self.gpu_transfer_collector.snapshots:
            gpu_csv = f"{output_prefix}_gpu_transfer.csv"
            gpu_snaps = self.gpu_transfer_collector.snapshots
            gpu_start = gpu_snaps[0].timestamp

            with open(gpu_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp_sec',
                    'relative_time_sec',
                    'gpu_id',
                    'tx_pci_mb_per_sec',
                    'rx_pci_mb_per_sec',
                    'cumulative_tx_gb',
                    'cumulative_rx_gb',
                ])

                cumulative_tx = 0.0
                cumulative_rx = 0.0
                for i, s in enumerate(gpu_snaps):
                    relative_time = s.timestamp - gpu_start
                    if i > 0:
                        dt = s.timestamp - gpu_snaps[i - 1].timestamp
                        cumulative_tx += s.tx_pci * dt / 1024  # MB to GB
                        cumulative_rx += s.rx_pci * dt / 1024

                    writer.writerow([
                        f"{s.timestamp:.3f}",
                        f"{relative_time:.3f}",
                        s.gpu_id,
                        f"{s.tx_pci:.2f}",
                        f"{s.rx_pci:.2f}",
                        f"{cumulative_tx:.4f}",
                        f"{cumulative_rx:.4f}",
                    ])

            print(f"Exported GPU transfer metrics to {gpu_csv}")

        # 3. Export client metrics (per-request stats)
        if client_metrics and len(client_metrics) > 0:
            client_csv = f"{output_prefix}_client_metrics.csv"
            sorted_metrics = sorted(client_metrics, key=lambda x: x.start_time_ms)
            first_start = sorted_metrics[0].start_time_ms

            with open(client_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'start_time_ms',
                    'relative_time_sec',
                    'ttft_ms',
                    'tpot_ms',
                    'latency_ms',
                    'input_num_turns',
                    'input_num_tokens',
                    'output_num_tokens',
                    'output_num_chunks',
                    'output_num_first_chunk_tokens',
                    'approx_cached_percent',
                    'conversation_id',
                    'client_id',
                    'interactivity_tok_per_sec',
                ])

                for m in sorted_metrics:
                    relative_time = (m.start_time_ms - first_start) / 1000.0
                    interactivity = 1000.0 / m.tpot_ms if m.tpot_ms > 0 else 0

                    writer.writerow([
                        f"{m.start_time_ms:.3f}",
                        f"{relative_time:.3f}",
                        f"{m.ttft_ms:.3f}",
                        f"{m.tpot_ms:.3f}",
                        f"{m.latency_ms:.3f}",
                        m.input_num_turns,
                        m.input_num_tokens,
                        m.output_num_tokens,
                        m.output_num_chunks,
                        m.output_num_first_chunk_tokens,
                        f"{m.approx_cached_percent:.2f}",
                        m.conversation_id,
                        m.client_id,
                        f"{interactivity:.2f}",
                    ])

            print(f"Exported client metrics to {client_csv}")
