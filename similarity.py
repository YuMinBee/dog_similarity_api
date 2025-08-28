# file name : similarity.py
# Date created : 2025-08-12
# Author : Yuminbee
import io, json, os, numpy as np, requests
from PIL import Image
import torch, clip

class DogSearcher:
    def __init__(self, emb_path: str, urls_path: str, device: str | None = None):
        self.emb = np.load(emb_path).astype("float32")           # (N, D)
        with open(urls_path, "r", encoding="utf-8") as f:
            self.urls = json.load(f)
        assert isinstance(self.urls, list) and len(self.urls) == self.emb.shape[0], \
            "embedding/URL Number Discrepancy"

        # Normalization
        n = np.linalg.norm(self.emb, axis=1, keepdims=True)
        self.emb = self.emb / np.clip(n, 1e-7, None)

        # CLIP load
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def _embed_pil(self, pil: Image.Image) -> np.ndarray:
        with torch.no_grad():
            x = self.preprocess(pil).unsqueeze(0).to(self.device)
            q = self.model.encode_image(x).float().cpu().numpy().reshape(-1).astype("float32")
        q /= max(float(np.linalg.norm(q)), 1e-7)
        return q

    def search_by_pil(self, pil: Image.Image, top_k: int = 5):
        q = self._embed_pil(pil)
        sims = self.emb @ q
        order = np.argsort(sims)[::-1][:max(top_k*3, top_k)]  # extra portion
        results = [{"idx": int(i), "sim": float(sims[i]), "url": self.urls[i]} for i in order]
        return results

    @staticmethod
    def is_alive(url: str, timeout=6) -> bool:
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True, headers=headers)
            if r.status_code >= 400 or not r.headers.get("content-type", "").startswith(("image/", "application/octet-stream")):
                r = requests.get(url, stream=True, timeout=timeout, headers=headers)
                r.close()
            return r.status_code < 400
        except Exception:
            return False

    def topk_alive(self, candidates, top_k: int):
        alive = []
        for c in candidates:
            if len(alive) >= top_k:
                break
            if self.is_alive(c["url"]):
                alive.append(c)
        return alive
