"""Descarga MovieLens 25M y registra fuente, versión, fecha y hash (rúbrica: fuente trazable)."""
import hashlib, urllib.request, zipfile, datetime
from .common import load_config, ROOT

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

def main():
    cfg = load_config()
    raw = ROOT / cfg["paths"]["raw"]
    if (raw / "ratings.csv").exists():
        print("Ya descargado:", raw); return
    raw.parent.mkdir(parents=True, exist_ok=True)
    zpath = raw.parent / "ml-25m.zip"
    print("Descargando", URL)
    urllib.request.urlretrieve(URL, zpath)
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw.parent)
    (raw.parent / "SOURCE.md").write_text(
        f"# Fuente\n- Dataset: MovieLens 25M (GroupLens)\n- URL: {URL}\n"
        f"- Fecha de descarga: {datetime.date.today()}\n- SHA256 del zip: {sha}\n", encoding="utf-8")
    zpath.unlink()
    print("Listo. Registro en", raw.parent / "SOURCE.md")

if __name__ == "__main__":
    main()
