"""Downloads the pretrained MediaPipe model files on first use."""
import urllib.request
from pathlib import Path


def ensure_file(url, path, status=None):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if status:
        status(f"Downloading {path.name} (first run only)...")
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path
