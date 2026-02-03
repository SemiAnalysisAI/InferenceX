"""
Metrics collector for vLLM server during benchmarks.
Polls /metrics endpoint and generates visualizations.
"""

import asyncio
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

import aiohttp
import matplotlib.pyplot as plt


@dataclass
class GpuTransferSnapshot:
    timestamp: float
    gpu_id: int = 0
    tx_pci: float = 0.0  # PCIe TX (MB/s)
    rx_pci: float = 0.0  # PCIe RX (MB/s)


class GpuTransferCollector:
    """Collects GPU transfer stats using nvidia-smi dmon."""

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

        # Start GPU transfer monitoring
        self.gpu_transfer_collector = GpuTransferCollector(
            gpu_id=self.gpu_id,
            poll_interval=max(1, int(self.poll_interval)),
        )
        self.gpu_transfer_collector.start()

    async def stop(self) -> None:
        """Stop metrics collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Stop GPU transfer monitoring
        if self.gpu_transfer_collector:
            self.gpu_transfer_collector.stop()

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
        num_rows = 5 if client_metrics else 3
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

        # 4. Throughput vs Time (tokens/sec)
        ax = axes[1, 1]
        throughputs = []
        for i in range(1, len(self.snapshots)):
            delta_gen = self.snapshots[i].generation_tokens - self.snapshots[i-1].generation_tokens
            delta_time = self.snapshots[i].timestamp - self.snapshots[i-1].timestamp
            if delta_time > 0:
                throughputs.append(delta_gen / delta_time)
            else:
                throughputs.append(0)
        ax.plot(times[1:], throughputs, 'orange', linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Generation Throughput")
        ax.grid(True, alpha=0.3)

        # 5. GPU Transfer TX over Time (scatter + rolling avg)
        ax = axes[2, 0]
        gpu_times = None
        if self.gpu_transfer_collector and len(self.gpu_transfer_collector.snapshots) > 1:
            gpu_snaps = self.gpu_transfer_collector.snapshots
            gpu_start = gpu_snaps[0].timestamp
            gpu_times = [(s.timestamp - gpu_start) for s in gpu_snaps]
            tx_pci = [s.tx_pci for s in gpu_snaps]  # Already in MB/s

            ax.scatter(gpu_times, tx_pci, alpha=0.3, s=5, c='blue', label='TX')
            # Rolling average
            gpu_window = min(50, len(tx_pci) // 10) if len(tx_pci) > 10 else 1
            if gpu_window > 1:
                rolling_tx = [
                    sum(tx_pci[max(0, i - gpu_window):i + 1]) / len(tx_pci[max(0, i - gpu_window):i + 1])
                    for i in range(len(tx_pci))
                ]
                ax.plot(gpu_times, rolling_tx, 'r-', linewidth=1.5, label=f'Rolling avg (n={gpu_window})')
            ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("TX Bandwidth (MB/s)")
        ax.set_title("GPU PCIe TX Over Time (GPU → Host)")
        ax.grid(True, alpha=0.3)

        # 6. GPU Transfer RX over Time (scatter + rolling avg)
        ax = axes[2, 1]
        if self.gpu_transfer_collector and len(self.gpu_transfer_collector.snapshots) > 1:
            gpu_snaps = self.gpu_transfer_collector.snapshots
            rx_pci = [s.rx_pci for s in gpu_snaps]  # Already in MB/s

            ax.scatter(gpu_times, rx_pci, alpha=0.3, s=5, c='blue', label='RX')
            # Rolling average
            gpu_window = min(50, len(rx_pci) // 10) if len(rx_pci) > 10 else 1
            if gpu_window > 1:
                rolling_rx = [
                    sum(rx_pci[max(0, i - gpu_window):i + 1]) / len(rx_pci[max(0, i - gpu_window):i + 1])
                    for i in range(len(rx_pci))
                ]
                ax.plot(gpu_times, rolling_rx, 'r-', linewidth=1.5, label=f'Rolling avg (n={gpu_window})')
            ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RX Bandwidth (MB/s)")
        ax.set_title("GPU PCIe RX Over Time (Host → GPU)")
        ax.grid(True, alpha=0.3)

        # 7 & 8. Client metrics plots (TTFT and Latency vs Time)
        if client_metrics and len(client_metrics) > 0:
            # Sort by start time
            sorted_metrics = sorted(client_metrics, key=lambda x: x.start_time_ms)
            # Convert to relative time (seconds from first request)
            first_start = sorted_metrics[0].start_time_ms
            request_times = [(m.start_time_ms - first_start) / 1000.0 for m in sorted_metrics]
            ttfts = [m.ttft_ms for m in sorted_metrics]
            latencies = [m.latency_ms for m in sorted_metrics]

            # 7. TTFT vs Time
            ax = axes[3, 0]
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

            # 8. Latency vs Time
            ax = axes[3, 1]
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

            # 9. Interactivity (1/TPOT = tokens/sec) vs Time
            ax = axes[4, 0]
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

            # Hide unused subplot
            axes[4, 1].axis('off')

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

        print("="*60 + "\n")
