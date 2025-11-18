from locust import HttpUser, task, between
import json
import random

SENTENCES = [
    "Hello world",
    "Machine learning is fascinating",
    "Optimization is key to performance",
    "Python is great for APIs",
    "Quantization reduces model size",
    "Redis makes things faster",
    "Batch processing improves throughput",
    "Latency vs Throughput trade-off",
    "Docker containers are useful",
    "Continuous integration is best practice"
]

class BatchUser(HttpUser):
    wait_time = between(0.1, 0.5) # Aggressive user

    @task(3)
    def batch_translate(self):
        """
        Sends a batch of 3-5 random sentences.
        This tests the Batch endpoint and Caching (since sentences repeat).
        """
        # Select random subset to simulate mixed hits/misses
        batch_size = random.randint(3, 5)
        batch_texts = random.choices(SENTENCES, k=batch_size)
        
        payload = {
            "texts": batch_texts,
            "source_lang": "en",
            "target_lang": "fr"
        }
        
        with self.client.post("/api/v2/translate/batch", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                hits = data['metrics']['cache_hits']
                # Optional: Verify performance logic
                # if hits > 0: print(f"Cache Hit! {hits}/{batch_size}")
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)
    def single_translate_legacy(self):
        # Keep hitting the old endpoint (if it exists) to measure degradation/comparison
        # Note: You need to merge the old endpoint into advanced_main.py if you want to test both simultaneously
        pass
