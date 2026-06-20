# Python Homework

This repository compares the performance of the original and optimized Python code.

## Project Structure

```text
Python-Homework/
├── src/
│   ├── data/
│   │   └── generate_dataset.py
│   ├── before/
│   │   ├── train.py
│   │   └── infer.py
│   │   └── ...
│   └── after/
│       ├── train.py
│       └── infer.py
│       └── ...
├── benchmark/
│   └── run_benchmark.py
└── results/
│  ├── benchmark_results.csv
│   └── graph/
│       ├── benchmark_train_average_time.png
│       └── benchmark_infer_average_time.png
└── report.pdf
```

## How to Run

```bash
git clone https://github.com/sangminlee-j/Python-Homework.git
cd Python-Homework/src/data
python3 generate_dataset.py
cd ../../benchmark
python3 run_benchmark.py
```

## Output

The benchmark script compares `before/` and `after/` versions of `train.py` and `infer.py`.

The results are saved in:

```text
results/benchmark_results.csv
results/graph/
```

The CSV file contains average runtime, standard deviation, average memory usage, and peak memory usage.

The graph directory contains runtime comparison figures for train and infer.
