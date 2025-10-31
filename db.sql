-- Enable pgvector
create extension if not exists vector;

-- Main table
create table if not exists public.astro_kb_chunks (
  id               text primary key,
  text             text not null,
  metadata         jsonb not null,
  entity_type      text not null,
  planet           text,
  house            int,
  sign             text,
  planet_a         text,
  planet_b         text,
  aspect           text,
  transit_by       text,
  target           text,
  transit_type     text,
  context          text not null,
  priority         int default 50,
  version          text default 'v1',
  embedding        vector(384) not null,
  created_at       timestamptz default now()
);

-- Secondary indexes
create index if not exists astro_kb_chunks_context_idx
  on public.astro_kb_chunks (context);
create index if not exists astro_kb_chunks_ph_idx
  on public.astro_kb_chunks (entity_type, planet, house);
create index if not exists astro_kb_chunks_aspect_idx
  on public.astro_kb_chunks (entity_type, planet_a, planet_b, aspect);
create index if not exists astro_kb_chunks_transit_idx
  on public.astro_kb_chunks (entity_type, transit_by, target, transit_type);
create index if not exists astro_kb_chunks_meta_gin
  on public.astro_kb_chunks using gin (metadata);

-- RPC: semantic search
create or replace function public.kb_semantic_search(
  q_embedding vector,
  p_context text,
  p_topk int default 24
)
returns table(
  id text,
  text text,
  metadata jsonb,
  entity_type text,
  planet text,
  house int,
  sign text,
  planet_a text,
  planet_b text,
  aspect text,
  transit_by text,
  target text,
  transit_type text,
  context text,
  priority int,
  version text,
  cosine_sim float
)
language sql stable as $$
  select
    id, text, metadata, entity_type, planet, house, sign, planet_a, planet_b, aspect,
    transit_by, target, transit_type, context, priority, version,
    1 - (embedding <=> q_embedding) as cosine_sim
  from public.astro_kb_chunks
  where context = p_context
  order by embedding <-> q_embedding
  limit p_topk;
$$;

-- Build IVF index (run after you have some rows)
-- drop index if exists astro_kb_chunks_embedding_ivf;
-- create index astro_kb_chunks_embedding_ivf
--   on public.astro_kb_chunks
--   using ivfflat (embedding vector_cosine_ops)
--   with (lists = 100);
