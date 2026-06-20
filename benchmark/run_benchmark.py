import csv
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "sampleproject_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARK_RUNS = 10
DATASET_SIZES_MB = [10, 40, 160]
VERSIONS = ["before", "after"]
SCRIPTS = ["train", "infer"]
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.02


BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BENCHMARK_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(SRC_DIR, "data")
RESULTS_DIR = os.path.abspath(os.path.join(BENCHMARK_DIR, "../results"))
GRAPH_DIR = os.path.join(RESULTS_DIR, "graph")
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark_results.csv")
TRAIN_FIGURE_PATH = os.path.join(GRAPH_DIR, "benchmark_train_average_time.png")
INFER_FIGURE_PATH = os.path.join(GRAPH_DIR, "benchmark_infer_average_time.png")


def dataset_path(size_mb):
    return os.path.join(DATA_DIR, f"dataset_{size_mb}mb.txt")


def ensure_inputs_exist():
    missing = [dataset_path(size_mb) for size_mb in DATASET_SIZES_MB if not os.path.exists(dataset_path(size_mb))]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing benchmark datasets:\n"
            f"{missing_list}\n\n"
            "Generate them first with:\n"
            "python3 src/data/generate_dataset.py"
        )


def rss_mb(pid):
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None

    output = result.stdout.strip()
    if not output:
        return None
    return int(output.splitlines()[0].strip()) / 1024


def run_measured(command, cwd):
    start = time.perf_counter()
    memory_samples = []

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    while process.poll() is None:
        memory = rss_mb(process.pid)
        if memory is not None:
            memory_samples.append(memory)
        time.sleep(MEMORY_SAMPLE_INTERVAL_SECONDS)

    memory = rss_mb(process.pid)
    if memory is not None:
        memory_samples.append(memory)

    _, stderr = process.communicate()
    elapsed = time.perf_counter() - start

    if process.returncode != 0:
        raise RuntimeError(
            "Benchmark command failed:\n"
            f"{' '.join(command)}\n\n"
            f"stderr:\n{stderr}"
        )

    average_memory = statistics.mean(memory_samples) if memory_samples else 0.0
    peak_memory = max(memory_samples) if memory_samples else 0.0
    return elapsed, average_memory, peak_memory


def command_for(version, script, size_mb):
    script_path = os.path.join(SRC_DIR, version, f"{script}.py")
    return [sys.executable, script_path, str(size_mb)]


def summarize(values):
    return {
        "avg_time_seconds": statistics.mean(item["time"] for item in values),
        "std_time_seconds": statistics.stdev(item["time"] for item in values) if len(values) > 1 else 0.0,
        "avg_memory_mb": statistics.mean(item["avg_memory"] for item in values),
        "peak_memory_mb": max(item["peak_memory"] for item in values),
    }


def run_benchmark():
    ensure_inputs_exist()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    rows = []
    for size_mb in DATASET_SIZES_MB:
        for version in VERSIONS:
            for script in SCRIPTS:
                runs = []
                for run_index in range(1, BENCHMARK_RUNS + 1):
                    print(f"Running {version}/{script}.py size={size_mb}MB run={run_index}/{BENCHMARK_RUNS}")
                    elapsed, average_memory, peak_memory = run_measured(
                        command_for(version, script, size_mb),
                        cwd=PROJECT_ROOT,
                    )
                    runs.append({
                        "time": elapsed,
                        "avg_memory": average_memory,
                        "peak_memory": peak_memory,
                    })

                summary = summarize(runs)
                rows.append({
                    "version": version,
                    "script": script,
                    "dataset_size_mb": size_mb,
                    "runs": BENCHMARK_RUNS,
                    **summary,
                })

    write_csv(rows)
    write_time_figures(rows)
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved train figure: {TRAIN_FIGURE_PATH}")
    print(f"Saved infer figure: {INFER_FIGURE_PATH}")


def write_csv(rows):
    by_case = {
        (row["dataset_size_mb"], row["script"], row["version"]): row
        for row in rows
    }
    fieldnames = [
        "Dataset Size (MB)",
        "Task",
        "Version",
        "Runtime Average (s)",
        "Runtime Std. Dev.",
        "Average Memory (MB)",
        "Peak Memory (MB)",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for script_index, script in enumerate(SCRIPTS):
            if script_index > 0:
                f.write("\n")
            for size_mb in DATASET_SIZES_MB:
                for version in VERSIONS:
                    row = by_case[(size_mb, script, version)]
                    writer.writerow({
                        "Dataset Size (MB)": f"{size_mb}MB",
                        "Task": script.capitalize(),
                        "Version": version.capitalize(),
                        "Runtime Average (s)": f"{row['avg_time_seconds']:.6f}",
                        "Runtime Std. Dev.": f"{row['std_time_seconds']:.6f}",
                        "Average Memory (MB)": f"{row['avg_memory_mb']:.6f}",
                        "Peak Memory (MB)": f"{row['peak_memory_mb']:.6f}",
                    })
        f.write("\n")
        f.write(f"Runs per case: {BENCHMARK_RUNS}\n")


def write_time_figures(rows):
    write_time_figure(rows, "train", TRAIN_FIGURE_PATH, "Average runtime (s)", 1.0)
    write_time_figure(rows, "infer", INFER_FIGURE_PATH, "Average runtime (ms)", 1000.0)


def write_time_figure(rows, script, figure_path, y_label, value_scale):
    values = {
        (row["dataset_size_mb"], row["version"]): row["avg_time_seconds"] * value_scale
        for row in rows
        if row["script"] == script
    }

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    x_positions = list(range(len(DATASET_SIZES_MB)))
    for version, marker in [("before", "o"), ("after", "s")]:
        y_values = [values.get((size_mb, version), 0.0) for size_mb in DATASET_SIZES_MB]
        ax.plot(
            x_positions,
            y_values,
            marker=marker,
            linewidth=2.5,
            markersize=8,
            label=f"{version}/{script}",
        )

    ax.set_title(f"{script.capitalize()} Average Execution Time")
    ax.set_xlabel("Dataset size (MB)")
    ax.set_ylabel(y_label)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{size_mb} MB" for size_mb in DATASET_SIZES_MB])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


if __name__ == "__main__":
    run_benchmark()
