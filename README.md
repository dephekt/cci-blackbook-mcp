# CCI Black Book MCP

MCP-only retrieval service for the CCI Black Book at `https://cci.ai.${DOMAIN}/mcp`.
The service returns bounded cited evidence packs; Codex or Claude Code synthesize the final answer.

This app is managed as a standalone uv project. Runtime dependencies are declared
in `pyproject.toml` and locked in `uv.lock`; the Docker image installs with
`uv sync --locked`. The image then force-reinstalls the locked
`onnxruntime-openvino` wheel with `uv pip` so it overrides FastEmbed's plain
`onnxruntime` dependency and exposes `OpenVINOExecutionProvider`.

## Private Data

Keep the PDF, index, cache, and token out of git:

```bash
sudo mkdir -p /mnt/data/cci-blackbook/{source,index,cache}
sudo cp "CCI Black Book.pdf" "/mnt/data/cci-blackbook/source/CCI Black Book.pdf"
install -d -m 700 cci/secrets
printf 'CCI_BLACKBOOK_MCP_TOKEN=%s\n' "$(openssl rand -hex 32)" > cci/secrets/cci.env
chmod 600 cci/secrets/cci.env
```

`cci/secrets/cci.env` must contain:

```dotenv
CCI_BLACKBOOK_MCP_TOKEN=replace-with-long-random-token
```

## Deployment

```bash
make cci-up
```

The compose service:

- joins the external `proxy` network for Pangolin/Newt discovery
- mounts `/dev/dri/renderD129` and adds render group `993`
- installs Intel GPU userspace drivers from `ppa:kobuk-team/intel-graphics`, matching the media-server host
- stores the SQLite index at `/mnt/data/cci-blackbook/index/`
- stores model/cache files at `/mnt/data/cci-blackbook/cache/`
- disables Pangolin SSO because MCP clients authenticate with bearer tokens

Prebuild or refresh the index after deployment:

```bash
docker --context media-server exec cci-blackbook cci-blackbook-ingest --force
```

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
