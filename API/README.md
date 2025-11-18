NMT API & Benchmarking Setup

1. Installation

Ensure you have Python 3.8+ installed. Install the necessary dependencies:

pip install fastapi uvicorn locust pydantic
# Optional: Install transformers and torch if you enable the real model in main.py
# pip install transformers torch sentencepiece


2. Running the API

Start the FastAPI server using Uvicorn. We disable the auto-reloader for benchmarking to get stable metrics.

uvicorn main:app --host 0.0.0.0 --port 8000


Observe the logs:
When you start the server, watch the console. You will see the "Cold Start" simulation:

Starting model load sequence...
Model loaded successfully in 2.5012 seconds.
API Ready. Total Startup Time: 2.5020 seconds.

3. Running the Benchmark (Locust)

With the API running in one terminal, open a new terminal to run the stress test.

locust -f locustfile.py


Open your browser and go to http://localhost:8089.

Set the Host to http://localhost:8000.

Set Number of users (e.g., 50) and Spawn rate (e.g., 5).

Click Start Swarming.

4. Metrics to Observe

API Console: You will see logs like Latency: 0.1502s for every request.

Locust UI:

RPS (Requests Per Second): Throughput.

Average Latency: The time taken for the request to complete.

Failures: Any non-200 responses.

5. Transitioning to Real ML Model

To use a real Neural Machine Translation model:

Open main.py.

Uncomment the transformers imports inside NMTModelWrapper.

Set use_real_model=True in the __init__ call:

nmt_model = NMTModelWrapper(use_real_model=True)
