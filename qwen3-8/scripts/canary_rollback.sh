#!/usr/bin/env bash
# Canary sequence: health → gen → streaming → tool → 32K → gateway switch.
# Any failure → exit, do NOT modify LiteLLM gateway backend.
#
# Usage: bash canary_rollback.sh [engine]
#   engine = vllm | sglang (default: vllm)
set -euo pipefail

ENGINE="${1:-vllm}"
BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
MODEL="${VLLM_MODEL:-qwen3.8-27b}"
RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_SCRIPT="${GATEWAY_SCRIPT:-/mnt/workspace/infra/studio_gateway.sh}"
GATEWAY_MODEL_NAME="${MODEL_NAME:-qwen3.8-27b}"

ok()   { printf '  OK   %s\n' "$*"; }
fail() { printf '  FAIL %s\n' "$*" >&2; exit 1; }

echo "=== Canary: $ENGINE → $MODEL @ $BASE_URL ==="

# Step 1: Health
curl -fsS --max-time 10 "$BASE_URL/health" >/dev/null || fail "health check"
ok "health 200"

# Step 2: Basic generation
python3 -c "
import json, urllib.request, os
api_key = os.environ.get('VLLM_API_KEY', '')
headers = {'Content-Type': 'application/json'}
if api_key:
    headers['Authorization'] = f'Bearer {api_key}'
body = json.dumps({'model': '$MODEL', 'messages': [{'role':'user','content':'Reply with OK'}], 'max_tokens': 16, 'temperature': 0}).encode()
req = urllib.request.Request('$BASE_URL/v1/chat/completions', data=body, headers=headers)
resp = json.load(urllib.request.urlopen(req, timeout=60))
content = resp['choices'][0]['message']['content']
assert 'OK' in content or 'ok' in content.lower(), f'unexpected: {content[:80]}'
print('  response:', content[:60])
" || fail "basic generation"
ok "basic generation"

# Step 3: Streaming
python3 -c "
import json, urllib.request, os
api_key = os.environ.get('VLLM_API_KEY', '')
headers = {'Content-Type': 'application/json'}
if api_key:
    headers['Authorization'] = f'Bearer {api_key}'
body = json.dumps({'model': '$MODEL', 'messages': [{'role':'user','content':'Count from 1 to 5.'}], 'max_tokens': 64, 'stream': True, 'temperature': 0}).encode()
req = urllib.request.Request('$BASE_URL/v1/chat/completions', data=body, headers=headers)
chunks = 0
for raw in urllib.request.urlopen(req, timeout=60):
    line = raw.decode().strip()
    if line.startswith('data: ') and line[6:] != '[DONE]':
        chunks += 1
assert chunks > 2, f'too few chunks: {chunks}'
print(f'  stream chunks: {chunks}')
" || fail "streaming"
ok "streaming"

# Step 4: Tool call
python3 -c "
import json, urllib.request, os, sys
sys.path.insert(0, '$RECIPE_ROOT/scripts/bench')
from bench_client import chat_completion
resp = chat_completion([{'role':'user','content':'Read file src/config.py'}], max_tokens=128, extra={'tools':[{'type':'function','function':{'name':'read_file','description':'Read file','parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}}}],'tool_choice':'auto'})
tc = resp.get('tool_calls',[])
assert tc and tc[0]['function']['name']=='read_file', f'no tool call: {tc}'
print(f'  tool call: {tc[0][\"function\"][\"name\"]}')
" || fail "tool call"
ok "tool call"

# Step 5: 32K context
python3 -c "
import sys
sys.path.insert(0, '$RECIPE_ROOT/scripts/bench')
from bench_client import chat_completion
prefix = 'Repository module: parser validates JSON with strict fields. ' * 2300  # ~32K
resp = chat_completion([{'role':'user','content':prefix + ' What does parser do? Reply in one sentence.'}], max_tokens=64)
assert resp['content'], 'empty response'
print(f'  32K response: {resp[\"content\"][:60]}')
" || fail "32K context"
ok "32K context"

# Step 6: Protocol fixtures (full suite)
python3 "$RECIPE_ROOT/scripts/bench/protocol_fixtures.py" --base-url "$BASE_URL" || fail "protocol fixtures"
ok "protocol fixtures"

echo ""
echo "=== Canary PASSED: $ENGINE is ready for gateway switch ==="
echo "To switch gateway: MODEL_NAME=$GATEWAY_MODEL_NAME bash $GATEWAY_SCRIPT restart"
echo "To verify: bash $GATEWAY_SCRIPT health"
