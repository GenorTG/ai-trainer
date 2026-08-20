#!/usr/bin/env bash
# Pre-commit hook: detect committed secrets
# Runs on every text file. Exits non-zero if suspicious patterns found.
set -eu

FILE="${1:?Usage: $0 <file>}"
PATTERNS=(
  'AKIA[0-9A-Z]{16}'                          # AWS access key
  'aws_secret_access_key'                      # AWS secret
  '-----BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY-----'  # private keys
  'ghp_[0-9a-zA-Z]{36}'                       # GitHub PAT
  'gho_[0-9a-zA-Z]{36}'                       # GitHub OAuth
  'github_pat_[0-9a-zA-Z_]{82}'               # GitHub fine-grained PAT
  'xox[baprs]-[0-9a-zA-Z-]{10,}'              # Slack tokens
  'sk-[0-9a-zA-Z]{20,}'                       # OpenAI-style API keys
  'sk_live_[0-9a-zA-Z]{24,}'                  # Stripe live keys
  'AIza[0-9A-Za-z_-]{35}'                     # Google API keys
  'ya29\.[0-9A-Za-z_-]+'                      # Google OAuth
  'supabase.*service_role'                     # Supabase service role
  'super-secret-genor-key'                    # Our test key
  'api[_-]?key.*=.*["'"'"'][0-9a-zA-Z]{20,}'  # Generic API key
)

for pattern in "${PATTERNS[@]}"; do
  if grep -E -q "$pattern" "$FILE" 2>/dev/null; then
    echo "❌ SECRET DETECTED in $FILE: pattern '$pattern'"
    echo "   Please use environment variables or .env files (gitignored)"
    exit 1
  fi
done

exit 0