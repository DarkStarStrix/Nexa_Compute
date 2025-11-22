import requests

class NexaClient:
    """Thin SDK wrapper around the Nexa FastAPI service.

    Usage example:
        client = NexaClient(api_key="YOUR_KEY", base_url="https://api.nexa.run")
        job = client.audit("s3://my-dataset")
        status = client.status(job["job_id"])
    """

    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def _post(self, path: str, payload: dict):
        url = f"{self.base_url}{path}"
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _get(self, path: str):
        url = f"{self.base_url}{path}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def audit(self, dataset_uri: str):
        return self._post("/audit", {"dataset_uri": dataset_uri})

    def distill(self, dataset_id: str, teacher: str = "openai/gpt-4o-mini"):
        return self._post("/distill", {"dataset_id": dataset_id, "teacher": teacher})

    def train(self, dataset_id: str, model: str = "Mistral-7B", epochs: int = 3):
        return self._post("/train", {"dataset_id": dataset_id, "model": model, "epochs": epochs})

    def evaluate(self, checkpoint_id: str):
        return self._post("/evaluate", {"checkpoint_id": checkpoint_id})

    def deploy(self, checkpoint_id: str):
        return self._post("/deploy", {"checkpoint_id": checkpoint_id})

    def status(self, job_id: str):
        return self._get(f"/status/{job_id}")
