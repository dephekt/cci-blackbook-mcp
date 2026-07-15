# CCI Black Book MCP

A small, self-contained **MCP server for semantic search over scanned / image-heavy
PDFs**. It was built for the *CCI Black Book* (a 505-page scanned grow manual), but it
works with **any PDF** — point it at your own document and go. It returns bounded,
cited evidence packs over HTTP; an MCP client (Claude Code, Codex, …) synthesizes the
final answer.

The interesting bit: most PDF-RAG tools index only the OCR text layer and silently drop
every diagram, chart, photo, and table. This one embeds **both** the text *and* a render
of each page, so the visual content is first-class in retrieval.

## How retrieval works

Three rankers, fused with weighted Reciprocal Rank Fusion:

- **text-dense** — [`voyage-context-4`](https://docs.voyageai.com/docs/contextualized-chunk-embeddings)
  contextualized chunk embeddings over the OCR text (one vector per chunk).
- **image-dense** — [`voyage-multimodal-3.5`](https://docs.voyageai.com/docs/multimodal-embeddings)
  over a PyMuPDF render of each page (one vector per page), so figures/photos/screenshots
  are retrievable **even where the OCR layer is empty**.
- **FTS/BM25** — exact keyword matches over the OCR text.

Every page is rendered; a conservative, logged blank-page filter drops lined "Notes:"
template pages (text-poor **and** ink-poor **and** color-poor) while keeping figure pages.
Results carry `unit_type` = `text` or `image`; image citations use ids like `p0042-img`.

Ingest **fails loud** if the embedding backend is unreachable (the prior index is left
intact); queries **degrade to FTS-only** if a dense space is unavailable.

## Quickstart (local, Docker)

```bash
git clone https://github.com/dephekt/cci-blackbook-mcp
cd cci-blackbook-mcp

cp secrets/cci.env.example secrets/cci.env       # a dev bearer token is pre-filled
mkdir -p data/source && cp /path/to/your.pdf data/source/document.pdf

# No API key needed — the offline "stub" backend exercises the whole pipeline:
CCI_EMBEDDING_BACKEND=stub docker compose up -d --build
docker compose exec cci-blackbook cci-blackbook-ingest --force

curl -s localhost:8000/healthz
```

Then point an MCP client at `http://127.0.0.1:8000/mcp` with the bearer token from
`secrets/cci.env` (see [Clients](#clients)). `docker compose up` auto-loads
`docker-compose.override.yml`, which publishes `127.0.0.1:8000` and mounts `./data` — no
reverse proxy or external network required.

### Real retrieval quality (Voyage)

The `stub` backend uses deterministic hash embeddings — fine for a smoke test, but not
semantically useful. For real results, use the default `voyage` backend:

1. Get a key at [voyageai.com](https://www.voyageai.com/) and put it in `secrets/cci.env`
   as `VOYAGE_API_KEY=...`.
2. Voyage **retains and may train on submitted data by default**. Turn on the one-way
   zero-retention opt-out in the Voyage dashboard, then set
   `CCI_VOYAGE_RETENTION_CONFIRMED=true` in `secrets/cci.env` (ingest is hard-gated on
   this — it refuses to send your PDF otherwise).
3. `docker compose up -d --build` (backend defaults to `voyage`) and
   `docker compose exec cci-blackbook cci-blackbook-ingest --force`.

Indexing a ~500-page book is roughly a **$0.60 one-time** cost (typically free-tier
covered). `cci-blackbook-ingest --smoke` validates Voyage connectivity with **synthetic
data only** and is safe to run any time.

## Quickstart (no Docker)

```bash
uv sync
export CCI_EMBEDDING_BACKEND=stub CCI_BLACKBOOK_MCP_TOKEN=dev-local-token \
       CCI_SOURCE_PDF=./data/source/document.pdf \
       CCI_INDEX_DIR=./data/index CCI_CACHE_DIR=./data/cache
uv run cci-blackbook-ingest --force
uv run cci-blackbook-mcp            # serves on http://0.0.0.0:8000
```

## Clients

Claude Code:

```bash
claude mcp add --transport http cci-blackbook http://127.0.0.1:8000/mcp \
  --header "Authorization: Bearer dev-local-token-change-me"
```

Codex:

```toml
[mcp_servers.cci_blackbook]
url = "http://127.0.0.1:8000/mcp"
bearer_token_env_var = "CCI_BLACKBOOK_MCP_TOKEN"
tool_timeout_sec = 120
```

## Tools

- `ask_blackbook(question, crop_context=None, facility_context=None, max_citations=6)`
- `blackbook_search(query, limit=10, mode="hybrid")` — modes: `hybrid` | `vector` | `fts` | `text` | `image`
- `blackbook_read_citation(chunk_id)` — accepts a text id (`p0042-c001`) or a page-image id (`p0042-img`)
- `blackbook_status()`

## Configuration

Everything is env-driven; the compose file sets sensible defaults. Common knobs:

| Variable | Default | Notes |
|---|---|---|
| `CCI_EMBEDDING_BACKEND` | `voyage` | or `stub` (offline, no key) |
| `CCI_SOURCE_PDF` | `/data/source/document.pdf` | the PDF to index |
| `VOYAGE_API_KEY` | — | required for the `voyage` backend |
| `CCI_VOYAGE_RETENTION_CONFIRMED` | `false` | must be `true` to ingest with `voyage` |
| `CCI_BLACKBOOK_MCP_TOKEN` | — | bearer token clients must send |
| `CCI_RENDER_DPI` | `100` | page render DPI for the image embeddings |
| `CCI_RRF_WEIGHT_IMAGE` | `2.0` | upweights the image ranker in fusion |

## Deploy behind a reverse proxy

`deploy/pangolin.yml` is an optional overlay (Pangolin/Newt shown; adapt for any proxy).
It drops the published host port and fronts the container's `:8000`:

```bash
DOMAIN=example.com CCI_DATA_DIR=/srv/cci-blackbook \
  docker compose -f docker-compose.yml -f deploy/pangolin.yml up -d --build
```

## Development

```bash
make test     # offline unit tests (no network, no PDF; uses the stub backend)
make lint     # ruff
```
