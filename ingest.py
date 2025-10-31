import os, re, glob, yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import numpy as np

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_NAME = "intfloat/multilingual-e5-small"
model = SentenceTransformer(MODEL_NAME)

def parse_md(path):
    txt = Path(path).read_text(encoding="utf-8")
    m = re.match(r"^---\n(.+?)\n---\n(.*)$", txt, flags=re.S|re.M)
    meta = yaml.safe_load(m.group(1)) if m else {}
    body = m.group(2).strip() if m else txt.strip()
    return meta, body

def chunk_text(body, max_chars=1200):
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf)+len(p) <= max_chars:
            buf += (("\n\n" if buf else "") + p)
        else:
            if buf:
                chunks.append(buf)
                buf=""
            if len(p) <= max_chars:
                chunks.append(p)
            else:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
    if buf:
        chunks.append(buf)
    return chunks

def embed(texts):
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embs.astype(np.float32)

def upsert_rows(rows):
    res = supabase.table("astro_kb_chunks").upsert(rows, on_conflict="id").execute()
    return res

def main(kb_dir="kb"):
    files = glob.glob(f"{kb_dir}/**/*.md", recursive=True)
    if not files:
        print("Nessun file .md trovato in", kb_dir)
        return
    batch = []
    for fp in files:
        meta, body = parse_md(fp)
        chunks = chunk_text(body)
        texts = [c.strip() for c in chunks]
        embs = embed(texts)
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
                "version": meta.get("version","v1"),
                "embedding": v.tolist()
            }
            batch.append(row)
    chunk_size = 200
    for i in range(0, len(batch), chunk_size):
        upsert_rows(batch[i:i+chunk_size])
    print(f"Upserted {len(batch)} chunks")

if __name__ == "__main__":
    main()
