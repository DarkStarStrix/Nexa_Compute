"""Load testing scripts."""

import concurrent.futures
import time
from typing import Dict

import requests

from nexa_compute.utils import get_logger

LOGGER = get_logger(__name__)


def stress_test_api(
    endpoint: str,
    concurrency: int = 10,
    duration_sec: int = 30,
    headers: Dict[str, str] = None,
) -> None:
    """Run concurrent requests against an endpoint."""
    start_time = time.time()
    success_count = 0
    fail_count = 0
    latencies = []
    
    def worker():
        try:
            req_start = time.time()
            resp = requests.get(endpoint, headers=headers, timeout=5)
            latency = time.time() - req_start
            if resp.status_code == 200:
                return True, latency
            return False, latency
        except Exception:
            return False, 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        # Submit initial batch
        for _ in range(concurrency * 2):
            futures.append(executor.submit(worker))
            
        while time.time() - start_time < duration_sec:
            done, not_done = concurrent.futures.wait(futures, timeout=0.1)
            for future in done:
                success, lat = future.result()
                if success:
                    success_count += 1
                    latencies.append(lat)
                else:
                    fail_count += 1
                
                # Schedule new task to maintain load
                futures.remove(future)
                if time.time() - start_time < duration_sec:
                    futures.append(executor.submit(worker))
            
            futures = list(not_done)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    rps = success_count / duration_sec
    
    LOGGER.info(
        "load_test_complete",
        extra={
            "rps": rps,
            "avg_latency": avg_latency,
            "success": success_count,
            "failed": fail_count,
        },
    )

if __name__ == "__main__":
    # Example usage
    stress_test_api("http://localhost:8000/health")
