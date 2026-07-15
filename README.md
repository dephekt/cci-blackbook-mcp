# CCI Black Book MCP

MCP-only retrieval service for the CCI Black Book at `https://cci.ai.${DOMAIN}/mcp`.
The service returns bounded cited evidence packs; Codex or Claude Code synthesize the final answer.

This app is managed as a standalone uv project. Runtime dependencies are declared
in `pyproject.toml` and locked in `uv.lock`; the Docker image installs with
`uv sync --locked`. PyMuPDF and Pillow ship self-contained wheels, so the image
needs no poppler or GPU system packages — only outbound HTTPS to Voyage.

## Embeddings (Voyage dual index)

The Black Book is a 505-page scanned book (an OCR text layer over page images),
so retrieval fuses **three** rankers with weighted Reciprocal Rank Fusion:

- **text-dense** — [`voyage-context-4`](https://docs.voyageai.com/docs/contextualized-chunk-embeddings)
  contextualized chunk embeddings over the OCR text chunks (one vector per chunk).
- **image-dense** — [`voyage-multimodal-3.5`](https://docs.voyageai.com/docs/multimodal-embeddings)
  over a render of each page (one vector per page), so diagrams, photos, tables,
  and UI screenshots become retrievable even where the OCR layer is empty.
- **FTS/BM25** — exact keyword matches over the OCR text (unchanged).

Every page is rendered with PyMuPDF; a conservative, logged blank-page filter drops
lined "Notes:" template pages (text-poor **and** ink-poor **and** color-poor) while
keeping figure/photo pages. `blackbook_search` supports `mode` =
`hybrid` (default) | `vector` | `fts` | `text` | `image`. Results carry
`unit_type` = `text` or `image`; image citations use ids like `p0042-img`.

Ingest **fails loud** if Voyage is unreachable (the prior index is left intact);
queries **degrade to FTS-only** if a dense space is unavailable at query time.
`CCI_EMBEDDING_BACKEND` accepts only `voyage` (default) or `stub` — an offline
blake2b hash provider used by the tests.

### Privacy

Ingest sends the book's OCR text **and** page images to Voyage's API. Voyage is
**retain-and-may-train by default** — turn on the one-way zero-retention opt-out in
the Voyage dashboard **before the first real ingest**. The build is hard-gated:
it refuses to run unless `CCI_VOYAGE_RETENTION_CONFIRMED=true`. The synthetic
`cci-blackbook-ingest --smoke` check sends only made-up data and is safe to run any time.

## Private Data

Keep the PDF, index, cache, and token out of git:

```bash
sudo mkdir -p /mnt/data/cci-blackbook/{source,index,cache}
sudo cp "CCI Black Book.pdf" "/mnt/data/cci-blackbook/source/CCI Black Book.pdf"
install -d -m 700 cci/secrets
printf 'CCI_BLACKBOOK_MCP_TOKEN=%s\n' "$(openssl rand -hex 32)" > cci/secrets/cci.env
chmod 600 cci/secrets/cci.env
```

Append the Voyage API key and the retention confirmation (both stay in the
gitignored secrets file, never in compose or logs):

```bash
printf 'VOYAGE_API_KEY=%s\n' "$(op read "op://Agents/CCI Embeddings Voyage/credential")" >> cci/secrets/cci.env
printf 'CCI_VOYAGE_RETENTION_CONFIRMED=%s\n' "true" >> cci/secrets/cci.env  # only after opting out
chmod 600 cci/secrets/cci.env
```

`cci/secrets/cci.env` must contain:

```dotenv
CCI_BLACKBOOK_MCP_TOKEN=replace-with-long-random-token
VOYAGE_API_KEY=replace-with-voyage-key
CCI_VOYAGE_RETENTION_CONFIRMED=true
```

## Deployment

```bash
make cci-up
```

The compose service:

- joins the external `proxy` network for Pangolin/Newt discovery
- stores the SQLite index at `/mnt/data/cci-blackbook/index/`
- stores cache files at `/mnt/data/cci-blackbook/cache/`
- disables Pangolin SSO because MCP clients authenticate with bearer tokens

Validate Voyage connectivity with synthetic data (safe pre-opt-out), then prebuild
or refresh the index after deployment:

```bash
docker --context media-server exec cci-blackbook cci-blackbook-ingest --smoke
docker --context media-server exec cci-blackbook cci-blackbook-ingest --force
```

`--force` needs `CCI_VOYAGE_RETENTION_CONFIRMED=true` in `cci.env`; it rebuilds the
whole index atomically (~$0.60 one-time, free-tier-covered) and exits non-zero on failure.

Run local focused tests:

```bash
uv run --project cci python -m unittest discover -s cci/tests -p 'test_*.py' -v
```

## Clients

Codex:

```toml
[mcp_servers.cci_blackbook]
url = "https://cci.ai.dephekt.net/mcp"
bearer_token_env_var = "CCI_BLACKBOOK_MCP_TOKEN"
tool_timeout_sec = 120
```

Claude Code:

```bash
claude mcp add --transport http cci-blackbook https://cci.ai.dephekt.net/mcp \
  --header "Authorization: Bearer $CCI_BLACKBOOK_MCP_TOKEN"
```

## Tools

- `ask_blackbook(question, crop_context=None, facility_context=None, max_citations=6)`
- `blackbook_search(query, limit=10, mode="hybrid")`
- `blackbook_read_citation(chunk_id)`
- `blackbook_status()`
