import requests
import time
import random

API_URL = "http://localhost:8000/api"

def create_test_data():
    print("Creating test data...")
    
    # 1. Register some workers
    workers = [
        {"worker_id": "worker-gpu-01", "hostname": "gpu-node-01", "gpu_type": "A100", "gpu_count": 8},
        {"worker_id": "worker-gpu-02", "hostname": "gpu-node-02", "gpu_type": "H100", "gpu_count": 4},
        {"worker_id": "worker-cpu-01", "hostname": "cpu-node-01", "gpu_type": "None", "gpu_count": 0},
    ]
    
    for w in workers:
        try:
            requests.post(f"{API_URL}/workers/register", json=w)
            print(f"Registered {w['worker_id']}")
            
            # Send a heartbeat
            requests.post(f"{API_URL}/workers/heartbeat", json={
                "worker_id": w['worker_id'],
                "status": random.choice(["idle", "busy", "idle"]),
                "current_job_id": f"job_{random.randint(1000, 9999)}" if random.random() > 0.5 else None
            })
        except Exception as e:
            print(f"Failed to register worker: {e}")

    # 2. Create some jobs
    job_types = ["generate", "audit", "distill", "train", "evaluate"]
    
    for i in range(10):
        jtype = random.choice(job_types)
        payload = {"domain": "test", "params": {"epochs": 10, "batch_size": 32}}
        
        try:
            requests.post(f"{API_URL}/jobs/{jtype}", json={"payload": payload})
            print(f"Created {jtype} job")
        except Exception as e:
            print(f"Failed to create job: {e}")

if __name__ == "__main__":
    create_test_data()
