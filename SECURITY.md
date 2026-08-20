# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| v0.1.x  | ✅ Active          |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
privately to **amy@smart-samurai.pl**. Do not file a public issue.

We will respond within 48 hours and provide a timeline for a fix.

## Security Practices

### Code
- `bandit` runs in CI on every commit
- `pip-audit` runs weekly to detect vulnerable packages
- All secrets must be in environment variables, never committed
- API key required for inference server (set `INFERENCE_API_KEY`)

### Models
- Models are sandboxed — system prompt restricts topics
- No tool execution on user-supplied URLs without confirmation
- RAG store is local-only by default
- Web search via MCP uses DuckDuckGo (no tracking)

### Infrastructure
- Inference server runs on internal network (Tailscale)
- HTTPS via reverse proxy (Nginx + Let's Encrypt)
- Rate limiting on public endpoint
- No PII in training data (synthetic persona only)

## Known Limitations

- **Prompt injection**: Models are vulnerable to adversarial prompts. The
  persona system prompt restricts topics, but determined attackers can bypass.
- **Hallucination**: Small models (14B) hallucinate facts. Always verify
  critical information.
- **No content filtering**: Server does not block harmful outputs by default.
  Operators should add upstream filtering if exposing publicly.

## Secrets Detection

Pre-commit hook `check-secrets` scans for common secret patterns:
- AWS access keys
- GitHub personal access tokens
- OpenAI API keys
- Stripe live keys
- Google API keys
- Generic `api_key=...` patterns

Never commit real secrets. Use environment variables and `.env` files (gitignored).

## Vulnerability Disclosure Timeline

1. **Day 0**: Report received
2. **Day 2**: Initial assessment
3. **Day 7**: Patch available
4. **Day 14**: Public disclosure (if coordinated)

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure).