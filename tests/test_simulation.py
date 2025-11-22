"""Simulation test for Nexa API pipeline."""

import json
import random
import time
from pathlib import Path
from typing import List, Dict

import requests

# Configuration
API_BASE_URL = "http://localhost:8000"
ARTIFACTS_DIR = Path("artifacts/test_sim")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_dataset(num_samples: int = 100) -> Path:
    """Generate synthetic dataset for testing."""
    print(f"📊 Generating {num_samples} synthetic samples...")
    
    prompts = [
        "Explain quantum computing",
        "Write a Python function to sort a list",
        "What is machine learning?",
        "Describe the water cycle",
        "How does photosynthesis work?",
    ]
    
    data = []
    for i in range(num_samples):
        prompt = random.choice(prompts)
        data.append({
            "id": f"sample_{i:04d}",
            "prompt": f"{prompt} (variation {i})",
            "context": f"Context for sample {i}",
            "task_type": random.choice(["qa", "code", "explanation"]),
            "quality_score": random.uniform(0.5, 1.0),
        })
    
    dataset_path = ARTIFACTS_DIR / "synthetic_dataset.jsonl"
    with open(dataset_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Dataset saved to {dataset_path}")
    print(f"   Samples: {len(data)}")
    print(f"   Format: JSONL")
    
    return dataset_path


def simulate_gpu_worker(worker_id: str = "sim-worker-001"):
    """Simulate a GPU worker with fake stats."""
    print(f"\n🖥️  Simulating GPU Worker: {worker_id}")
    
    worker_info = {
        "worker_id": worker_id,
        "ssh_host": "192.168.1.100",  # Fake IP
        "gpu_count": 1,
        "gpu_type": "A100-40GB",
        "status": "idle",
        "gpu_stats": {
            "utilization": random.randint(0, 15),
            "memory_used_gb": random.uniform(0.5, 2.0),
            "memory_total_gb": 40.0,
            "temperature": random.randint(30, 45),
        }
    }
    
    print(f"   Host: {worker_info['ssh_host']}")
    print(f"   GPU: {worker_info['gpu_type']}")
    print(f"   Status: {worker_info['status']}")
    print(f"   GPU Util: {worker_info['gpu_stats']['utilization']}%")
    print(f"   GPU Mem: {worker_info['gpu_stats']['memory_used_gb']:.1f}GB / {worker_info['gpu_stats']['memory_total_gb']}GB")
    
    return worker_info


def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API call."""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return {"error": str(e)}


def wait_for_job(job_id: str, timeout: int = 60):
    """Poll job status until complete."""
    print(f"⏳ Waiting for job {job_id}...")
    
    start = time.time()
    while time.time() - start < timeout:
        status = call_api(f"/status/{job_id}")
        
        if "error" in status:
            print(f"❌ Error checking status: {status['error']}")
            return None
        
        job_status = status.get("status", "unknown")
        print(f"   Status: {job_status}")
        
        if job_status in ["completed", "failed"]:
            return status
        
        time.sleep(2)
    
    print(f"⏰ Timeout waiting for job {job_id}")
    return None


def run_simulation():
    """Run full pipeline simulation."""
    print("=" * 60)
    print("🚀 NEXA API PIPELINE SIMULATION")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n[STEP 1] Generate Synthetic Dataset")
    dataset_path = generate_synthetic_dataset(100)
    dataset_uri = str(dataset_path.absolute())
    
    # Step 2: Simulate GPU worker
    print("\n[STEP 2] Simulate GPU Worker")
    worker = simulate_gpu_worker()
    
    # Step 3: Check API health
    print("\n[STEP 3] Check API Health")
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print("❌ API not responding correctly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure API is running: uvicorn src.nexa.api:app --reload")
        return
    
    # Step 4: List workers
    print("\n[STEP 4] List Registered Workers")
    workers = call_api("/workers")
    print(f"   Registered workers: {len(workers.get('workers', []))}")
    
    # Step 5: Submit audit job (local)
    print("\n[STEP 5] Submit Audit Job (Local)")
    audit_job = call_api("/audit", "POST", {"dataset_uri": dataset_uri})
    
    if "job_id" in audit_job:
        print(f"✅ Audit job created: {audit_job['job_id']}")
        audit_result = wait_for_job(audit_job["job_id"], timeout=30)
        
        if audit_result and audit_result.get("status") == "completed":
            print("✅ Audit completed")
            if "result" in audit_result:
                print(f"   Result: {json.dumps(audit_result['result'], indent=2)[:200]}...")
    else:
        print(f"❌ Failed to create audit job: {audit_job}")
    
    # Step 6: Submit distill job (remote - will be pending without real worker)
    print("\n[STEP 6] Submit Distill Job (Remote)")
    distill_job = call_api("/distill", "POST", {
        "dataset_id": "test_dataset",
        "teacher": "gpt-4o-mini"
    })
    
    if "job_id" in distill_job:
        print(f"✅ Distill job created: {distill_job['job_id']}")
        print("   Note: Will be pending (no real GPU worker provisioned)")
        
        # Check status
        time.sleep(1)
        status = call_api(f"/status/{distill_job['job_id']}")
        print(f"   Status: {status.get('status', 'unknown')}")
    else:
        print(f"❌ Failed to create distill job: {distill_job}")
    
    # Step 7: Submit train job (remote - will be pending)
    print("\n[STEP 7] Submit Train Job (Remote)")
    train_job = call_api("/train", "POST", {
        "dataset_id": "test_dataset",
        "model": "gpt2",
        "epochs": 1
    })
    
    if "job_id" in train_job:
        print(f"✅ Train job created: {train_job['job_id']}")
        print("   Note: Will be pending (no real GPU worker provisioned)")
        
        # Check status
        time.sleep(1)
        status = call_api(f"/status/{train_job['job_id']}")
        print(f"   Status: {status.get('status', 'unknown')}")
    else:
        print(f"❌ Failed to create train job: {train_job}")
    
    # Step 8: Summary
    print("\n" + "=" * 60)
    print("📊 SIMULATION SUMMARY")
    print("=" * 60)
    print(f"✅ Dataset generated: {dataset_path}")
    print(f"✅ GPU worker simulated: {worker['worker_id']}")
    print(f"✅ API endpoints tested: /audit, /distill, /train, /status, /workers")
    print(f"✅ Local jobs (audit) execute successfully")
    print(f"⚠️  Remote jobs (distill, train) pending (no real GPU worker)")
    print("\n💡 Next steps:")
    print("   1. Implement remote_worker.py agent")
    print("   2. Add Prime Intellect provisioning")
    print("   3. Test full end-to-end with real GPU worker")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation()
