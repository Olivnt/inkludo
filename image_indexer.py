# image_indexer.py
import io, os, uuid, requests
from PIL import Image
import torch
from torchvision import transforms
from sentence_transformers import SentenceTransformer

# Simple CLIP-like encoder (e.g., clip-ViT-B-32)
MODEL = os.environ.get("IMAGE_EMB_MODEL", "clip-ViT-B-32")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = SentenceTransformer(MODEL, device=_device)

_pre = transforms.Compose([transforms.Resize((224,224))])  # minimal; sbert handles preprocessing

def image_vector_from_url(url, headers):
    bin = requests.get(url, headers=headers).content
    img = Image.open(io.BytesIO(bin)).convert("RGB")
    # sentence-transformers CLIP model expects PIL directly:
    vec = _model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec
