# AstroBot KB Starter (Supabase + pgvector, free)

Contenuto:
- KB di esempio in Markdown (kb/)
- db.sql (schema, RPC, index)
- ingest.py (ingest + embeddings e5-small)
- rag.py (ricerca ibrida + prompt builder)
- requirements.txt
- .env.example

Setup rapido:
1) In Supabase: crea progetto, apri SQL Editor e incolla db.sql. Esegui.
2) Copia Project URL e service_role key da Settings -> API. Metti in .env.
3) pip install -r requirements.txt
4) python ingest.py
5) (opzionale) attiva l'indice IVF in db.sql dopo qualche riga inserita.
6) Integra le funzioni di rag.py nel tuo FastAPI.
