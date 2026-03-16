"""
temporary profiler — run alongside the download pipeline to sample system metrics.

usage:
	python -m proteus.data.downloads.proteus_dataset.profile_run [--pid PID] [--interval 0.5] [--out profile.jsonl]

if --pid is omitted, monitors the current process tree (useful when run in-process).
can also be imported and used as a context manager:

	with RunProfiler(path="profile.jsonl", interval=0.5):
		run_pipeline(...)
"""

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path

import psutil


def _snapshot(proc: psutil.Process) -> dict:
	"""collect one sample of system + process metrics."""
	now = time.monotonic()

	# process tree metrics
	try:
		children = proc.children(recursive=True)
	except psutil.NoSuchProcess:
		children = []

	all_procs = [proc] + children
	tree_pids = {p.pid for p in all_procs}
	total_threads = 0
	proc_rss_mb = 0.0
	proc_cpu = 0.0
	proc_io_read_mb = 0.0
	proc_io_write_mb = 0.0
	for p in all_procs:
		try:
			mem = p.memory_info()
			proc_rss_mb += mem.rss / (1024 * 1024)
			proc_cpu += p.cpu_percent(interval=0)
			total_threads += p.num_threads()
			try:
				io = p.io_counters()
				proc_io_read_mb += io.read_bytes / (1024 * 1024)
				proc_io_write_mb += io.write_bytes / (1024 * 1024)
			except (psutil.AccessDenied, AttributeError):
				pass
		except (psutil.NoSuchProcess, psutil.AccessDenied):
			continue

	# top non-pipeline processes by CPU
	top_other = []
	try:
		for p in psutil.process_iter(["pid", "name", "cpu_percent"]):
			if p.pid in tree_pids:
				continue
			cpu = p.info.get("cpu_percent") or 0.0
			if cpu > 5.0:
				top_other.append({"pid": p.pid, "name": p.info["name"], "cpu_pct": round(cpu, 1)})
		top_other.sort(key=lambda x: x["cpu_pct"], reverse=True)
		top_other = top_other[:5]
	except Exception:
		pass

	# system-wide metrics
	cpu_pct = psutil.cpu_percent(interval=0, percpu=False)
	cpu_per_core = psutil.cpu_percent(interval=0, percpu=True)
	vm = psutil.virtual_memory()
	net = psutil.net_io_counters()

	return {
		"t": round(now, 3),
		# system
		"sys_cpu_pct": cpu_pct,
		"sys_cpu_per_core": cpu_per_core,
		"sys_mem_used_mb": round(vm.used / (1024 * 1024), 1),
		"sys_mem_pct": vm.percent,
		"net_bytes_sent": net.bytes_sent,
		"net_bytes_recv": net.bytes_recv,
		# process tree
		"proc_threads": total_threads,
		"proc_rss_mb": round(proc_rss_mb, 1),
		"proc_cpu_pct": round(proc_cpu, 1),
		"proc_io_read_mb": round(proc_io_read_mb, 1),
		"proc_io_write_mb": round(proc_io_write_mb, 1),
		"top_other_procs": top_other,
	}


class RunProfiler:
	"""samples system metrics in a background thread, writes jsonl."""

	def __init__(
		self,
		path: str | Path = "profile.jsonl",
		interval: float = 0.5,
		pid: int | None = None,
	):
		self.path = Path(path)
		self.interval = interval
		self.pid = pid or os.getpid()
		self._stop = threading.Event()
		self._thread: threading.Thread | None = None
		self._phase = "init"
		self._phase_lock = threading.Lock()

	def set_phase(self, name: str):
		with self._phase_lock:
			self._phase = name

	def _run(self):
		proc = psutil.Process(self.pid)
		# prime cpu_percent counters
		proc.cpu_percent(interval=0)
		for child in proc.children(recursive=True):
			try:
				child.cpu_percent(interval=0)
			except (psutil.NoSuchProcess, psutil.AccessDenied):
				pass
		psutil.cpu_percent(interval=0, percpu=True)
		time.sleep(0.1)

		t0 = None
		prev_net_sent = 0
		prev_net_recv = 0

		with open(self.path, "w") as f:
			while not self._stop.is_set():
				try:
					snap = _snapshot(proc)
				except psutil.NoSuchProcess:
					break

				if t0 is None:
					t0 = snap["t"]
					prev_net_sent = snap["net_bytes_sent"]
					prev_net_recv = snap["net_bytes_recv"]

				elapsed = snap["t"] - t0
				# compute network rates
				net_sent_rate = (snap["net_bytes_sent"] - prev_net_sent) / self.interval / (1024 * 1024)
				net_recv_rate = (snap["net_bytes_recv"] - prev_net_recv) / self.interval / (1024 * 1024)
				prev_net_sent = snap["net_bytes_sent"]
				prev_net_recv = snap["net_bytes_recv"]

				with self._phase_lock:
					phase = self._phase

				row = {
					"elapsed_s": round(elapsed, 2),
					"phase": phase,
					"sys_cpu_pct": snap["sys_cpu_pct"],
					"cores_active": sum(1 for c in snap["sys_cpu_per_core"] if c > 50),
					"cores_total": len(snap["sys_cpu_per_core"]),
					"sys_mem_pct": snap["sys_mem_pct"],
					"sys_mem_used_mb": snap["sys_mem_used_mb"],
					"proc_rss_mb": snap["proc_rss_mb"],
					"proc_cpu_pct": snap["proc_cpu_pct"],
					"proc_threads": snap["proc_threads"],
					"proc_io_read_mb": snap["proc_io_read_mb"],
					"proc_io_write_mb": snap["proc_io_write_mb"],
					"net_recv_mbps": round(net_recv_rate, 2),
					"net_send_mbps": round(net_sent_rate, 2),
					"top_other_procs": snap["top_other_procs"],
				}
				f.write(json.dumps(row) + "\n")
				f.flush()

				self._stop.wait(self.interval)

	def start(self):
		self._thread = threading.Thread(target=self._run, daemon=True, name="run-profiler")
		self._thread.start()

	def stop(self):
		self._stop.set()
		if self._thread:
			self._thread.join(timeout=5)

	def __enter__(self):
		self.start()
		return self

	def __exit__(self, *exc):
		self.stop()


def print_summary(path: Path):
	"""print a quick summary of the profiling data."""
	rows = []
	with open(path) as f:
		for line in f:
			rows.append(json.loads(line))

	if not rows:
		print("no data collected")
		return

	duration = rows[-1]["elapsed_s"]
	print(f"\n{'=' * 60}")
	print(f"  profiling summary — {duration:.1f}s total, {len(rows)} samples")
	print(f"{'=' * 60}")

	def _stats(key):
		vals = [r[key] for r in rows]
		return min(vals), sum(vals) / len(vals), max(vals)

	for label, key in [
		("sys cpu %", "sys_cpu_pct"),
		("proc cpu %", "proc_cpu_pct"),
		("proc threads", "proc_threads"),
		("proc rss mb", "proc_rss_mb"),
		("sys mem %", "sys_mem_pct"),
		("proc io read mb", "proc_io_read_mb"),
		("proc io write mb", "proc_io_write_mb"),
		("net recv mb/s", "net_recv_mbps"),
		("net send mb/s", "net_send_mbps"),
	]:
		lo, avg, hi = _stats(key)
		print(f"  {label:<16s}  min={lo:>8.1f}  avg={avg:>8.1f}  max={hi:>8.1f}")

	cores_active = [r["cores_active"] for r in rows]
	cores_total = rows[0]["cores_total"]
	print(f"  {'cores >50%':<16s}  min={min(cores_active):>8d}  avg={sum(cores_active)/len(cores_active):>8.1f}  max={max(cores_active):>8d}  (of {cores_total})")

	# phase breakdown
	phases = {}
	for r in rows:
		phase = r.get("phase", "unknown")
		if phase not in phases:
			phases[phase] = {"count": 0, "first": r["elapsed_s"], "last": r["elapsed_s"]}
		phases[phase]["count"] += 1
		phases[phase]["last"] = r["elapsed_s"]
	print(f"\n  phases:")
	for phase, info in phases.items():
		print(f"    {phase:<20s}  {info['first']:>6.1f}s - {info['last']:>6.1f}s  ({info['count']} samples)")

	# top other processes summary
	other_seen: dict[str, float] = {}
	for r in rows:
		for p in r.get("top_other_procs", []):
			name = p["name"]
			other_seen[name] = max(other_seen.get(name, 0), p["cpu_pct"])
	if other_seen:
		print(f"\n  top non-pipeline processes (peak cpu%):")
		for name, peak in sorted(other_seen.items(), key=lambda x: x[1], reverse=True)[:10]:
			print(f"    {name:<30s}  peak={peak:>6.1f}%")

	print(f"{'=' * 60}\n")


def print_stage_summary(path: Path):
	"""print summary of per-entry stage timing from stage_times.jsonl."""
	rows = []
	with open(path) as f:
		for line in f:
			rows.append(json.loads(line))

	if not rows:
		print("no stage timing data")
		return

	print(f"\n{'=' * 60}")
	print(f"  stage timing summary — {len(rows)} entries")
	print(f"{'=' * 60}")

	stages = ["fetch", "cpu_wait", "write_lock_wait", "write", "total"]
	for stage in stages:
		vals = [r[stage] for r in rows]
		vals.sort()
		p95 = vals[int(len(vals) * 0.95)] if len(vals) >= 20 else vals[-1]
		print(f"  {stage:<18s}  min={min(vals):>8.4f}  avg={sum(vals)/len(vals):>8.4f}  p95={p95:>8.4f}  max={max(vals):>8.4f}")

	raw_vals = [r["raw_kb"] for r in rows]
	blob_vals = [r["blob_kb"] for r in rows]
	print(f"\n  {'raw_kb':<18s}  min={min(raw_vals):>8.1f}  avg={sum(raw_vals)/len(raw_vals):>8.1f}  max={max(raw_vals):>8.1f}")
	print(f"  {'blob_kb':<18s}  min={min(blob_vals):>8.1f}  avg={sum(blob_vals)/len(blob_vals):>8.1f}  max={max(blob_vals):>8.1f}")

	pool_vals = [r["pool_active"] for r in rows if r.get("pool_active", -1) >= 0]
	if pool_vals:
		print(f"  {'pool_active':<18s}  min={min(pool_vals):>8d}  avg={sum(pool_vals)/len(pool_vals):>8.1f}  max={max(pool_vals):>8d}")

	# concurrency timeline — how many entries were in each stage per second
	if rows:
		max_wall = max(r["wall"] + r["total"] for r in rows)
		buckets = int(max_wall) + 1
		in_fetch = [0] * buckets
		in_cpu = [0] * buckets
		in_write = [0] * buckets
		for r in rows:
			t0 = r["wall"]
			t_fetch_end = t0 + r["fetch"]
			t_cpu_end = t_fetch_end + r["cpu_wait"]
			t_write_end = t_cpu_end + r["write_lock_wait"] + r["write"]
			for s in range(int(t0), min(int(t_fetch_end) + 1, buckets)):
				in_fetch[s] += 1
			for s in range(int(t_fetch_end), min(int(t_cpu_end) + 1, buckets)):
				in_cpu[s] += 1
			for s in range(int(t_cpu_end), min(int(t_write_end) + 1, buckets)):
				in_write[s] += 1

		print(f"\n  concurrency timeline (entries in stage per second):")
		print(f"  {'sec':<6s}  {'fetch':>6s}  {'cpu':>6s}  {'write':>6s}")
		for s in range(buckets):
			if in_fetch[s] or in_cpu[s] or in_write[s]:
				print(f"  {s:<6d}  {in_fetch[s]:>6d}  {in_cpu[s]:>6d}  {in_write[s]:>6d}")

	print(f"{'=' * 60}\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="profile a running process")
	parser.add_argument("--pid", type=int, default=None, help="pid to monitor (default: self)")
	parser.add_argument("--interval", type=float, default=0.5, help="sample interval in seconds")
	parser.add_argument("--out", type=str, default="profile.jsonl", help="output file")
	parser.add_argument("--summary", type=str, default=None, help="just print summary of existing file")
	parser.add_argument("--stage-summary", type=str, default=None, help="print summary of stage_times.jsonl")
	args = parser.parse_args()

	if args.summary:
		print_summary(Path(args.summary))
		if args.stage_summary:
			print_stage_summary(Path(args.stage_summary))
		sys.exit(0)

	if args.stage_summary:
		print_stage_summary(Path(args.stage_summary))
		sys.exit(0)

	pid = args.pid or os.getpid()
	print(f"profiling pid {pid}, interval {args.interval}s, writing to {args.out}")
	print("press ctrl+c to stop\n")

	profiler = RunProfiler(path=args.out, interval=args.interval, pid=pid)
	profiler.start()
	try:
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		pass
	profiler.stop()
	print_summary(Path(args.out))
