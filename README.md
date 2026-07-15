# CCI Black Book MCP

An MCP server that does semantic search over the **CCI Black Book** — a large, scanned
grow manual — and returns bounded, cited evidence packs for an MCP client (Claude Code,
Codex, …) to answer from.

Scanned books are mostly *pictures*. Naive PDF-RAG indexes only the OCR text layer and
loses every chart, diagram, and photo. This server embeds **both** the text and a render
of each page, so the visual content is first-class in retrieval.

## Requirements

- A **[Voyage AI](https://www.voyageai.com/) API key** — the embeddings are the whole
  point, so this is required.
- The source PDF (the CCI Black Book, or another scanned/image-heavy PDF).
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
Results carry `unit_type` = `text` or `image`; image citations use ids like `p0042-img`.
Ingest **fails loud** if Voyage is unreachable (the prior index is left intact); queries
**degrade to FTS-only** if a dense space is unavailable.

## Quickstart

```bash
git clone https://github.com/dephekt/cci-blackbook-mcp
cd cci-blackbook-mcp

# 1. Secrets: a bearer token clients present, plus your Voyage key.
cp secrets/cci.env.example secrets/cci.env
#    then edit secrets/cci.env:
#      VOYAGE_API_KEY=...                     (from https://www.voyageai.com/)
#      CCI_VOYAGE_RETENTION_CONFIRMED=true    (after opting out — see Privacy)

# 2. Drop the PDF in.
mkdir -p data/source && cp "/path/to/CCI Black Book.pdf" data/source/document.pdf

# 3. Build, start, and index (~$0.60 one-time, ~8 min for a ~500-page book).
docker compose up -d --build
docker compose exec cci-blackbook cci-blackbook-ingest --force
```

`docker compose up` publishes the MCP on `127.0.0.1:8000` and serves the PDF/index from
`./data` — no reverse proxy or external network required. Point your client at
`http://127.0.0.1:8000/mcp` (see [Clients](#clients)).

Verify Voyage connectivity first with `cci-blackbook-ingest --smoke` — it sends only
synthetic data, so it's safe and essentially free.

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

- `ask_blackbook(question, crop_context=None, facility_context=None, max_citations=6)`
- `blackbook_search(query, limit=10, mode="hybrid")` — modes: `hybrid` | `vector` | `fts` | `text` | `image`
- `blackbook_read_citation(chunk_id)` — accepts a text id (`p0042-c001`) or a page-image id (`p0042-img`)
- `blackbook_status()`

## Configuration

Env-driven; the compose file sets sensible defaults. The ones you'll actually set:

| Variable | Notes |
|---|---|
| `VOYAGE_API_KEY` | **required** |
| `CCI_VOYAGE_RETENTION_CONFIRMED` | must be `true` to ingest (see Privacy) |
| `CCI_BLACKBOOK_MCP_TOKEN` | bearer token clients must send |
| `CCI_SOURCE_PDF` | defaults to `/data/source/document.pdf` |
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
