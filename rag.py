import os
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ["SUPABASE_SERVICE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

model = SentenceTransformer("intfloat/multilingual-e5-small")

def build_query_ctx(mode:str, placements:Dict[str,Any], transits:List[Dict[str,Any]]|None=None):
    q = {"must": {"context": mode}, "should": []}
    for planet, info in placements.items():
        if planet == "aspects":
            continue
        if isinstance(info, dict) and "house" in info:
            q["should"].append({"entity_type":"planet_house","planet":planet,"house":info["house"]})
        if isinstance(info, dict) and "sign" in info:
            q["should"].append({"entity_type":"planet_sign","planet":planet,"sign":info["sign"]})
    for asp in placements.get("aspects", []):
        q["should"].append({"entity_type":"aspect","planet_a":asp["a"],"planet_b":asp["b"],"aspect":asp["type"]})
    for t in (transits or []):
        q["should"].append({"entity_type":"transit","transit_by":t["by"],"target":t["target"],"transit_type":t["type"]})
    return q

def query_text(ctx:Dict[str,Any])->str:
    parts = [f"context:{ctx['must']['context']}"]
    for s in ctx["should"]:
        parts.append(" ".join(f"{k}:{v}" for k,v in s.items()))
    return " | ".join(parts)

def search_topk(ctx:Dict[str,Any], top_k:int=12):
    qtxt = query_text(ctx)
    qemb = model.encode([qtxt], normalize_embeddings=True)[0].astype("float32").tolist()
    res = sb.rpc("kb_semantic_search", {
        "q_embedding": qemb,
        "p_context": ctx["must"]["context"],
        "p_topk": top_k*2
    }).execute()
    rows = res.data or []
    def soft_match_score(row):
        score = row.get("cosine_sim", 0.0)
        for s in ctx["should"]:
            score += sum(0.02 for k,v in s.items() if v and str(row.get(k)) == str(v))
        score += (row.get("priority",50) / 1000.0)
        return score
    rows.sort(key=soft_match_score, reverse=True)
    return rows[:top_k]

def to_bullets(rows):
    out = []
    for r in rows:
        label = r["id"]
        note  = r["text"].strip().replace("\n"," ")
        out.append(f"- [{label}] {note}")
    return "\n".join(out)

def build_synthesis_prompt(ctx, rows):
    ctx_str = query_text(ctx)
    bullets = to_bullets(rows)
    prompt = (
        "Ruolo: Astrologo analitico e pratico.\n\n"
        "Istruzioni:\n"
        "- Riformula i concetti con parole tue, niente copia testuale dal corpus.\n"
        "- Integra i punti coerenti; separa (+) opportunita e (-) criticita.\n"
        "- Se mancano elementi, dichiara l'incertezza.\n"
        "- Tono: chiaro, non deterministico, orientato all'azione.\n"
        "- Evita profezie; usa probabilita, condizioni e contesti.\n"
        f"Contesto astro: {ctx_str}\n\n"
        "Note di base (da rielaborare, non citare):\n"
        f"{bullets}\n\n"
        "Output richiesto:\n"
        "1) breve paragrafo introduttivo (2-3 frasi)\n"
        "2) 3-6 bullet (+)\n"
        "3) 3-6 bullet (-)\n"
        "4) 1-2 suggerimenti pratici specifici\n"
    )
    return prompt

if __name__ == "__main__":
    placements = {
      "Mars":  {"house": 1, "sign": "Aries"},
      "Venus": {"house": 4, "sign": "Cancer"},
      "aspects": [{"a":"Mars","b":"Venus","type":"square","orb":2.0}]
    }
    ctx = build_query_ctx("natal", placements, transits=None)
    from pprint import pprint
    hits = search_topk(ctx, top_k=6)
    pprint(hits[:2])
