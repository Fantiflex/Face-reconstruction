from pathlib import Path
import shutil
import re

src = Path("data/raw/cfd/images") ## DOSSIER CONTENANT TOUS LES SOUS-DOSSIERS DE LA CFD" 
dst = Path("data/raw/cfd/neutral_CFD") ## LÀ OÙ ON VEUT LA CFD NEUTRE (UN SEUL DOSSIER C'EST PLUS PRATIQUE))
dst.mkdir(parents=True, exist_ok=True)

def is_neutral(filename):
    stem = Path(filename).stem.lower()
    return (
        "neutral" in stem
        or bool(re.search(r"(?:^|[-_\s])n$", stem))
    )

count = 0

for image in src.rglob("*"):
    if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        if is_neutral(image.name):
            shutil.copy2(image, dst / image.name)
            count += 1

print(f"{count} neutral images retained in {dst}")
