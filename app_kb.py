# app_kb.py
from fastapi import APIRouter
from typing import Dict, Any, List
from rag import build_query_ctx, search_topk, build_synthesis_prompt

# Sostituisci con la tua funzione LLM
def interpreta_groq(prompt: str) -> str:
    return f"[DEMO] Sintesi pronta su prompt di {len(prompt)} caratteri."

kb_router = APIRouter()

@router.post("/interpreta-natal")
def interpreta_natal(placements: Dict[str, Any]):
    ctx = build_query_ctx("natal", placements, transits=None)
    rows = search_topk(ctx, top_k=12)
    prompt = build_synthesis_prompt(ctx, rows)
    testo = interpreta_groq(prompt)
    return {"ok": True, "interpretazione": testo, "chunks_usati": [r["id"] for r in rows]}

@router.post("/interpreta-transit")
def interpreta_transit(payload: Dict[str, Any]):
    placements: Dict[str, Any] = payload.get("placements", {})
    transits: List[Dict[str, Any]] = payload.get("transits", [])
    ctx = build_query_ctx("transit", placements, transits=transits)
    rows = search_topk(ctx, top_k=12)
    prompt = build_synthesis_prompt(ctx, rows)
    testo = interpreta_groq(prompt)
    return {"ok": True, "interpretazione": testo, "chunks_usati": [r["id"] for r in rows]}
