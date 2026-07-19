# CCI Black Book MCP

An MCP server that does semantic search over **one or more grow manuals** (e.g. the CCI
Black Book) — a corpus of scanned/image-heavy PDFs — and returns bounded, cited evidence
packs for an MCP client (Claude Code, Codex, …) to answer from. Every hit is labeled with
the book it came from, and you can scope a query to specific book(s).

Scanned books are mostly *pictures*. Naive PDF-RAG indexes only the OCR text layer and
loses every chart, diagram, and photo. This server embeds **both** the text and a render
of each page, so the visual content is first-class in retrieval.

## Requirements

- A **[Voyage AI](https://www.voyageai.com/) API key** — the embeddings are the whole
  point, so this is required.
- One or more source PDFs (the CCI Black Book and/or other grow manuals).
- Docker, or Python 3.12 + [uv](https://docs.astral.sh/uv/).

## How retrieval works

Three rankers, fused with weighted Reciprocal Rank Fusion:

- **text-dense** — [`voyage-context-4`](https://docs.voyageai.com/docs/contextualized-chunk-embeddings)
  contextualized chunk embeddings over the OCR text (one vector per chunk).
- **image-dense** — [`voyage-multimodal-3.5`](https://docs.voyageai.com/docs/multimodal-embeddings)
  over a PyMuPDF render of each page (one vector per page), so figures/photos/charts are
  retrievable **even where the OCR layer is empty**.
- **FTS/BM25** — exact keyword matches over the OCR text.

Every page is rendered; a conservative, logged blank-page filter drops lined "Notes:"
template pages (text-poor **and** ink-poor **and** color-poor) while keeping figure pages.
Results carry `unit_type` = `text` or `image` plus a `source_id`/`source_title`; unit ids
are namespaced per book (e.g. `cci-black-book:p0042-img`). Ingest **fails loud** if Voyage
is unreachable (the prior index is left intact); queries **degrade to FTS-only** if a dense
space is unavailable.

## Quickstart

```bash
git clone https://github.com/dephekt/cci-blackbook-mcp
cd cci-blackbook-mcp

# 1. Secrets: a bearer token clients present, plus your Voyage key.
cp secrets/cci.env.example secrets/cci.env
#    then edit secrets/cci.env:
#      VOYAGE_API_KEY=...                     (from https://www.voyageai.com/)
#      CCI_VOYAGE_RETENTION_CONFIRMED=true    (after opting out — see Privacy)

# 2. Drop one or more PDFs into data/source/. Each file's real name becomes its
#    source id + title (e.g. "CCI Black Book.pdf" -> cci-black-book / "CCI Black Book").
mkdir -p data/source
cp "/path/to/CCI Black Book.pdf" "/path/to/Aroya Guide to Drying.pdf" data/source/

# 3. Build, start, and index (~$0.60 initially, ~8 min for a ~500-page book).
docker compose up -d --build
docker compose exec cci-blackbook cci-blackbook-ingest
```

`docker compose up` publishes the MCP on `127.0.0.1:8000` and serves the PDF/index from
`./data` — no reverse proxy or external network required. Point your client at
`http://127.0.0.1:8000/mcp` (see [Clients](#clients)).

Verify Voyage connectivity first with `cci-blackbook-ingest --smoke` — it sends only
synthetic data, so it's safe and essentially free.

### Incremental ingestion and schema upgrades

Normal ingestion is source-incremental. It hashes each discovered PDF, compares its stored text
and image pipeline fingerprints, and embeds only added or modified books. Removed books are
deleted locally, and an unchanged corpus makes no Voyage calls. Every affected book rebuilds both
embedding spaces in this first implementation.

Use `make ingest` for normal refreshes. `make ingest-force` is an explicit full rebuild that
regenerates every embedding and may consume Voyage allowance or incur charges.

Schema-v3 indexes are never migrated or modified in place. Upgrade one offline:

```bash
docker compose stop cci-blackbook
docker compose run --rm cci-blackbook cci-blackbook-ingest --force
docker compose up -d cci-blackbook
```

The forced command builds and validates a separate schema-v4 database before atomically replacing
the legacy file. Keep the MCP stopped for the complete one-off command.

### Privacy

Ingest sends the book's OCR text **and** page images to Voyage's API. Voyage **retains and
may train on submitted data by default** — turn on the one-way zero-retention opt-out in
the Voyage dashboard, then set `CCI_VOYAGE_RETENTION_CONFIRMED=true`. Ingest is hard-gated
on this flag and refuses to send anything otherwise.

## Clients

Claude Code:

```bash
claude mcp add --transport http cci-blackbook http://127.0.0.1:8000/mcp \
  --header "Authorization: Bearer $CCI_BLACKBOOK_MCP_TOKEN"
```

Codex:

```toml
[mcp_servers.cci_blackbook]
url = "http://127.0.0.1:8000/mcp"
bearer_token_env_var = "CCI_BLACKBOOK_MCP_TOKEN"
tool_timeout_sec = 120
```

## Tools

- `ask_blackbook(question, crop_context=None, facility_context=None, max_citations=6, sources=None)`
- `blackbook_search(query, limit=10, mode="hybrid", sources=None)` — modes: `hybrid` | `vector` | `fts` | `text` | `image`
- `blackbook_read_citation(chunk_id)` — accepts a namespaced text id (`cci-black-book:p0042-c001`) or page-image id (`cci-black-book:p0042-img`)
- `blackbook_status()` — lists every indexed book with per-book counts

`sources` scopes a query to one book id or a list of them (default: all); ids come from
`blackbook_status()`. Each result carries `source_id`/`source_title` (the book) — distinct
from the result's `sources` field, which lists the rankers (fts/text_dense/image_dense)
that surfaced it.

## Configuration

Env-driven; the compose file sets sensible defaults. The ones you'll actually set:

| Variable | Notes |
|---|---|
| `VOYAGE_API_KEY` | **required** |
| `CCI_VOYAGE_RETENTION_CONFIRMED` | must be `true` to ingest (see Privacy) |
| `CCI_BLACKBOOK_MCP_TOKEN` | bearer token clients must send |
| `CCI_SOURCE_DIR` | directory of PDFs to index (default `/data/source`) |
| `CCI_RENDER_DPI` | `100` — page render DPI for the image embeddings |
| `CCI_RRF_WEIGHT_IMAGE` | `2.0` — upweights the image ranker in fusion |

## Deploy behind a reverse proxy

`deploy/pangolin.yml` is an optional overlay (Pangolin/Newt shown; adapt for any proxy).
It drops the published host port and fronts the container's `:8000`:

```bash
DOMAIN=example.com CCI_DATA_DIR=/srv/cci-blackbook \
  docker compose -f docker-compose.yml -f deploy/pangolin.yml up -d --build
```

## Development

```bash
make test   # offline unit tests — deterministic fixtures, no network or API key
make lint   # ruff
```
