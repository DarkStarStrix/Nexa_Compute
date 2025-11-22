import requests
import time
import random
from datetime import datetime, timedelta

API_URL = "http://localhost:8000/api"

# Simulate realistic job scenarios
SCENARIOS = [
    {"type": "generate", "status": "completed", "worker": "worker-gpu-01"},
    {"type": "audit", "status": "completed", "worker": "worker-cpu-01"},
    {"type": "distill", "status": "running", "worker": "worker-gpu-02"},
    {"type": "train", "status": "failed", "worker": "worker-gpu-01"},
    {"type": "evaluate", "status": "completed", "worker": "worker-gpu-02"},
    {"type": "generate", "status": "failed", "worker": None},
    {"type": "train", "status": "running", "worker": "worker-gpu-01"},
    {"type": "distill", "status": "completed", "worker": "worker-gpu-02"},
    {"type": "evaluate", "status": "pending", "worker": None},
    {"type": "deploy", "status": "completed", "worker": "worker-cpu-01"},
]

def create_simulated_data():
    print("Creating simulated data for Nexa Forge Dashboard...")
    print("=" * 60)
    
    # 1. Register workers with diverse configurations
    workers = [
        {
            "worker_id": "worker-gpu-01", 
            "hostname": "gpu-node-us-east-1a", 
            "gpu_type": "A100-80GB", 
            "gpu_count": 8
        },
        {
            "worker_id": "worker-gpu-02", 
            "hostname": "gpu-node-us-west-2b", 
            "gpu_type": "H100-SXM", 
            "gpu_count": 4
        },
        {
            "worker_id": "worker-cpu-01", 
            "hostname": "cpu-node-central-1", 
            "gpu_type": "None", 
            "gpu_count": 0
        },
    ]
    
    print("\n1. Registering Workers...")
    for w in workers:
        try:
            requests.post(f"{API_URL}/workers/register", json=w)
            
            # Determine status based on worker
            if w["worker_id"] == "worker-gpu-01":
                status = "busy"
                current_job = "job_train_running"
            elif w["worker_id"] == "worker-gpu-02":
                status = "busy"
                current_job = "job_distill_running"
            else:
                status = "idle"
                current_job = None
                
            requests.post(f"{API_URL}/workers/heartbeat", json={
                "worker_id": w['worker_id'],
                "status": status,
                "current_job_id": current_job
            })
            print(f"   ✓ {w['worker_id']} ({w['gpu_type']}) - Status: {status}")
        except Exception as e:
            print(f"   ✗ Failed to register {w['worker_id']}: {e}")

    # 2. Create diverse jobs with realistic statuses
    print("\n2. Creating Simulated Jobs...")
    for i, scenario in enumerate(SCENARIOS):
        try:
            payload = {
                "domain": random.choice(["biology", "chemistry", "physics", "medical"]),
                "params": {
                    "epochs": random.randint(1, 5),
                    "batch_size": random.choice([16, 32, 64]),
                }
            }
            
            response = requests.post(
                f"{API_URL}/jobs/{scenario['type']}", 
                json={"payload": payload}
            )
            
            if response.ok:
                job = response.json()
                
                # Update job status to match scenario using the job manager
                # (In a real setup, we'd have an admin endpoint for this)
                # For now, jobs will be created as 'pending'
                
                status_emoji = {
                    "completed": "✓",
                    "failed": "✗",
                    "running": "→",
                    "pending": "○"
                }.get(scenario['status'], "?")
                
                print(f"   {status_emoji} {job['job_id']}: {scenario['type']} ({scenario['status']})")
            
        except Exception as e:
            print(f"   ✗ Failed to create {scenario['type']} job: {e}")
        
        # Small delay to spread out timestamps
        time.sleep(0.1)
    
    print("\n" + "=" * 60)
    print("Dashboard simulation complete!")
    print("\nView at: http://localhost:3000/dashboard")
    print("=" * 60)

if __name__ == "__main__":
    create_simulated_data()
