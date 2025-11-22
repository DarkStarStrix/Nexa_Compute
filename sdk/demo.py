"""
Test script for Nexa Forge SDK.
Demonstrates the complete workflow: API key generation and SDK usage.
"""

import sys
sys.path.insert(0, 'sdk')

from nexa_forge import NexaForgeClient

def main():
    print("=" * 60)
    print("Nexa Forge SDK Demo")
    print("=" * 60)
    
    # For demo purposes, we'll use the SDK without an API key
    # In production, users would get their key from the dashboard
    print("\n1. Initializing client...")
    client = NexaForgeClient(api_url="http://localhost:8000/api")
    
    print("\n2. Generating data...")
    job1 = client.generate(
        domain="medical_imaging",
        num_samples=100
    )
    print(f"   ✓ Data generation job created: {job1['job_id']}")
    print(f"   Status: {job1['status']}")
    
    print("\n3. Submitting distillation job...")
    job2 = client.distill(
        teacher_model="gpt-4",
        student_model="llama-3-8b",
        dataset_uri="s3://my-bucket/training-data.parquet"
    )
    print(f"   ✓ Distillation job created: {job2['job_id']}")
    
    print("\n4. Submitting training job...")
    job3 = client.train(
        model_id="llama-3-8b",
        dataset_uri="s3://my-bucket/finetuning-data.parquet",
        epochs=3
    )
    print(f"   ✓ Training job created: {job3['job_id']}")
    
    print("\n5. Submitting evaluation job...")
    job4 = client.evaluate(
        model_id="my-finetuned-model",
        benchmark="mmlu"
    )
    print(f"   ✓ Evaluation job created: {job4['job_id']}")
    
    print("\n6. Listing all jobs...")
    jobs = client.list_jobs(limit=5)
    print(f"   Found {len(jobs)} recent jobs:")
    for job in jobs[:5]:
        print(f"   - {job['job_id']}: {job['job_type']} ({job['status']})")
    
    print("\n7. Checking job status...")
    status = client.get_job(job1['job_id'])
    print(f"   Job {status['job_id']}: {status['status']}")
    
    print("\n" + "=" * 60)
    print("Demo complete! 🎉")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Visit http://localhost:3000/dashboard/settings")
    print("  2. Generate an API key")
    print("  3. Use it with: client = NexaForgeClient(api_key='YOUR_KEY')")
    print("=" * 60)

if __name__ == "__main__":
    main()
