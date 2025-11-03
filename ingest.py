import os, re, glob, yaml
from pathlib import Path
from typing import List
import numpy as np
from supabase import create_client, Client

# fastembed import robusto
try:
    from fastembed import TextEmbedding
except ImportError:
    from fastembed.embedding import TextEmbedding  # fallback

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBED = TextEmbedding(model_name=EMBEDDING_MODEL)  # 384-d

KB_DIR = os.environ.get("KB_DIR", "kb")
REQUIRED = ["id", "entity_type", "context"]

def parse_md(path):
    txt = Path(path).read_text(encoding="utf-8")
    m = re.match(r"^---\n(.+?)\n---\n(.*)$", txt, flags=re.S | re.M)
    meta = yaml.safe_load(m.group(1)) if m else {}
    body = m.group(2).strip() if m else txt.strip()
    return meta, body

def validate_meta(meta, src):
    miss = [k for k in REQUIRED if not meta.get(k)]
    if miss:
        raise ValueError(f"Missing {miss} in front-matter for {src}")

def chunk_text(body, max_chars=1200):
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) <= max_chars:
            buf += (("\n\n" if buf else "") + p)
        else:
            if buf: chunks.append(buf); buf = ""
            if len(p) <= max_chars:
                chunks.append(p)
            else:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
    if buf: chunks.append(buf)
    return chunks

def embed(texts: List[str]) -> np.ndarray:
    arr = list(EMBED.embed(texts))
    return np.vstack(arr).astype(np.float32)  # (N,384)

def upsert_rows(rows):
    return supabase.table("astro_kb_chunks").upsert(rows, on_conflict="id").execute()

def main():
    files = glob.glob(f"{KB_DIR}/**/*.md", recursive=True)
    print(f"Found {len(files)} .md under {KB_DIR}/")
    if not files:
        return

    batch = []
    for fp in files:
        meta, body = parse_md(fp)
        validate_meta(meta, fp)
        chunks = chunk_text(body)
        texts = [c.strip() for c in chunks]
        if not texts: continue
        embs = embed(texts)  # (n,384)
        for i, (t, v) in enumerate(zip(texts, embs)):
            rid = f"{meta.get('id','')}-{i:02d}"
            row = {
                "id": rid,
                "text": t,
                "metadata": meta,
                "entity_type": meta.get("entity_type"),
                "planet": meta.get("planet"),
                "house": meta.get("house"),
                "sign": meta.get("sign"),
                "planet_a": meta.get("planet_a"),
                "planet_b": meta.get("planet_b"),
                "aspect": meta.get("aspect"),
                "transit_by": meta.get("transit_by"),
                "target": meta.get("target"),
                "transit_type": meta.get("transit_type"),
                "context": meta.get("context", "natal"),
                "priority": meta.get("priority", 50),
                "version": meta.get("version", "v1"),
                "embedding": v.tolist()
            }
            batch.append(row)

    print(f"Upserting {len(batch)} chunks…")
    for i in range(0, len(batch), 200):
        upsert_rows(batch[i:i+200])
    print("Done.")

if __name__ == "__main__":
    main()
