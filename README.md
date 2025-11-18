# High-Performance NMT API (FastAPI + ONNX + Redis)

A comprehensive performance engineering project demonstrating the optimization of a Neural Machine Translation (NMT) API. This repository tracks the evolution from a standard PyTorch/FastAPI implementation (Baseline) to a highly optimized architecture using ONNX Runtime Quantization, Redis Caching, and Batch Inference.

## 🚀 Project Overview

The goal of this project is to demonstrate how specific engineering decisions impact Latency, Throughput, and Cold Start times in ML deployment.

### Phase 1: The Baseline
Framework: FastAPI + Hugging Face transformers (PyTorch).
Inference: standard float32 eager execution.
Architecture: Simple synchronous endpoint (with thread pooling).
Metrics: High cold start time, higher latency per token.

### Phase 2: The Optimization
Quantization: Converted model to INT8 using ONNX Runtime, reducing model size by ~75%.
Caching: Implemented a Redis layer to cache frequently requested translations (O(1) lookup).
Batching: Added a vectorised endpoint to process multiple sentences in a single matrix operation.
Concurrency: Non-blocking I/O for caching mixed with thread-pooled CPU-bound inference.

## 🛠️ Tech Stack
Language: Python 3.8+
API: FastAPI, Uvicorn
ML Inference: Hugging Face Transformers, Optimum, ONNX Runtime
Data Store: Redis (Async)
Benchmarking: Locust

## 📦 Installation
Clone the repository:
git clone [https://github.com/yourusername/nmt-performance-api.git](https://github.com/yourusername/nmt-performance-api.git)
cd nmt-performance-api

Install Dependencies:
pip install fastapi uvicorn locust pydantic transformers torch sentencepiece
pip install optimum[onnxruntime] redis


(Optional) Run Redis:
If you want to test Phase 2 caching, you need a Redis instance.
docker run --name my-redis -p 6379:6379 -d redis


📊 Phase 1: Baseline Architecture
The baseline represents a standard implementation found in many tutorials.
Start the API:
uvicorn main:app --host 0.0.0.0 --port 8000

Observe the logs to see the simulated "Cold Start" time (~2.5s).
Run Benchmarks:
Open a new terminal:
locust -f locustfile.py


Go to http://localhost:8089.
Target Host: http://localhost:8000.
Simulate 50 users / 5 spawn rate.
⚡ Phase 2: Optimized Architecture
This phase introduces advanced techniques to reduce latency and increase RPS.
1. Quantize the Model
Run the optimization script to download the Hugging Face model, export it to ONNX, and apply INT8 quantization.
python optimize_model.py


Check the console output to see the model size reduction (approx 300MB -> 75MB).
2. Start the Optimized API
This uses advanced_main.py, which loads the ONNX model and connects to Redis.
uvicorn advanced_main:app --host 0.0.0.0 --port 8000


3. Run Batch Benchmarks
Use the specialized load test that sends batch requests and repeats sentences to trigger cache hits.
locust -f benchmark_v2.py


📈 Performance Results
Metric
Baseline (Phase 1)
Optimized (Phase 2)
Improvement
Model Size
~300 MB
~75 MB
-75%
Cold Start
~2.5s
~0.8s
3x Faster
Inference (P95)
~150ms
~40ms
~3.7x Faster
Throughput
~10-15 RPS
~500+ RPS
Massive Gain

Note: Throughput gains are heavily driven by Batching + Redis Cache hits.
📂 Project Structure
.
├── main.py              # Phase 1: Baseline API
├── locustfile.py        # Phase 1: Standard Load Test
├── optimize_model.py    # Utility: Converts PyTorch -> ONNX INT8
├── advanced_main.py     # Phase 2: Optimized API (Batching + Redis + ONNX)
├── benchmark_v2.py      # Phase 2: Batch/Cache Load Test
└── onnx_models/         # Generated artifacts (Gitignored)


