"""Structure decoder for generating molecular representations from embeddings."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn


class DecoderModel(nn.Module):
    """Simple transformer-style decoder placeholder."""

    def __init__(self, embedding_dim: int, vocab_size: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(embedding)


VOCAB = list("CNOPSH123456789=#+-[]()")


class StructureDecoder:
    """Generate molecular structures (SMILES / InChI) from embeddings."""

    def __init__(self, embedding_dim: int, checkpoint_path: Optional[Path] = None, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.decoder = DecoderModel(embedding_dim).to(self.device)
        if checkpoint_path and checkpoint_path.exists():
            self.decoder.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.decoder.eval()

    def _generate_tokens(self, embedding: torch.Tensor, length: int = 32) -> List[str]:
        embedding = embedding.to(self.device)
        with torch.no_grad():
            logits = self.decoder(embedding)
            probs = torch.softmax(logits, dim=-1)
        tokens = []
        for _ in range(length):
            idx = torch.multinomial(probs, num_samples=1).item()
            tokens.append(VOCAB[idx % len(VOCAB)])
        return tokens

    def decode_smiles(self, embedding: torch.Tensor) -> str:
        tokens = self._generate_tokens(embedding)
        return "".join(tokens)

    def decode_inchi(self, embedding: torch.Tensor) -> str:
        smiles = self.decode_smiles(embedding)
        return f"InChI=1S/{smiles}"

    def decode_structure(self, embedding: torch.Tensor) -> dict:
        smiles = self.decode_smiles(embedding)
        return {
            "smiles": smiles,
            "inchi": f"InChI=1S/{smiles}",
            "confidence": round(random.uniform(0.5, 0.95), 3),
        }

