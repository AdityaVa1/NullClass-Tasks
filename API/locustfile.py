from locust import HttpUser, task, between
import json

class NMTUser(HttpUser):
    # Wait time between tasks (simulating a real user thinking time)
    # Here: between 0.5 and 2 seconds
    wait_time = between(0.5, 2)

    @task
    def translate_text(self):
        payload = {
            "text": "Hello, this is a test sentence for benchmarking.",
            "source_lang": "en",
            "target_lang": "fr"
        }
        
        headers = {"Content-Type": "application/json"}
        
        with self.client.post("/api/v1/translate", data=json.dumps(payload), headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                # You can parse the response here if needed
                # json_response = response.json()
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def health_check(self):
        # Occasionally check health endpoint (weighted lower than translate)
        self.client.get("/health")
