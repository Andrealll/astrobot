# rag.py
import os
from typing import Dict, Any, List

from supabase import create_client

# Import robusto per fastembed (supporta più versioni)
try:
    from fastembed import TextEmbedding
except ImportError:
    from fastembed.embedding import TextEmbedding  # fallback

# ---- Supabase client ----
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_SERVICE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- Embedding model (384-d) ----
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
_MODEL: TextEmbedding | None = None

def _get_model() -> TextEmbedding:
    global _MODEL
    if _MODEL is None:
        _MODEL = TextEmbedding(model_name=_EMBEDDING_MODEL)  # 384-dim
    return _MODEL

def _embed_one(text: str) -> list:
    model = _get_model()
    # fastembed.embed restituisce un iteratore di np.ndarray
    vec = next(model.embed([text]))
    return vec.astype("float32").tolist()  # (384,)

# ---- Query building ----
def build_query_ctx(mode: str, placements: Dict[str, Any], transits: List[Dict[str, Any]] | None = None):
    q = {"must": {"context": mode}, "should": []}
    for planet, info in placements.items():
        if planet == "aspects":
            continue
        if isinstance(info, dict) and "house" in info:
            q["should"].append({"entity_type": "planet_house", "planet": planet, "house": info["house"]})
        if isinstance(info, dict) and "sign" in info:
            q["should"].append({"entity_type": "planet_sign", "planet": planet, "sign": info["sign"]})
    for asp in placements.get("aspects", []):
        q["should"].append({"entity_type": "aspect", "planet_a": asp["a"], "planet_b": asp["b"], "aspect": asp["type"]})
    for t in (transits or []):
        q["should"].append({"entity_type": "transit", "transit_by": t["by"], "target": t["target"], "transit_type": t["type"]})
    return q

def query_text(ctx: Dict[str, Any]) -> str:
    parts = [f"context:{ctx['must']['context']}"]
    for s in ctx["should"]:
        parts.append(" ".join(f"{k}:{v}" for k, v in s.items()))
    return " | ".join(parts)

def _soft_match_score(row, ctx):
    score = row.get("priority", 50) / 100.0
    for s in ctx["should"]:
        score += sum(0.05 for k, v in s.items() if v and str(row.get(k)) == str(v))
    return score

# ---- Vector search + rerank leggero ----
def search_topk(ctx: Dict[str, Any], top_k: int = 12):
    qtxt = query_text(ctx)
    qemb = _embed_one(qtxt)
    res = sb.rpc(
        "kb_semantic_search",
        {"q_embedding": qemb, "p_context": ctx["must"]["context"], "p_topk": top_k * 2},
    ).execute()
    rows = res.data or []
    rows.sort(key=lambda r: _soft_match_score(r, ctx), reverse=True)
    return rows[:top_k]

def to_bullets(rows):
    out = []
    for r in rows:
        note = (r.get("text") or "").strip().replace("\n", " ")
        out.append(f"- [{r.get('id','?')}] {note}")
    return "\n".join(out)

def build_synthesis_prompt(ctx, rows):
    ctx_str = query_text(ctx)
    bullets = to_bullets(rows)
    return (
        "Ruolo: Astrologo analitico e pratico.\n\n"
        "Istruzioni:\n"
        "- Riformula i concetti con parole tue, niente copia testuale dal corpus.\n"
        "- Integra i punti coerenti; separa (+) opportunita e (-) criticita.\n"
        "- Se mancano elementi, dichiara l'incertezza.\n"
        "- Tono: chiaro, non deterministico, orientato all'azione.\n"
        f"Contesto astro: {ctx_str}\n\n"
        "Note di base (da rielaborare, non citare):\n"
        f"{bullets}\n\n"
        "Output richiesto:\n"
        "1) breve paragrafo introduttivo (2-3 frasi)\n"
        "2) 3-6 bullet (+)\n"
        "3) 3-6 bullet (-)\n"
        "4) 1-2 suggerimenti pratici specifici\n"
    )
