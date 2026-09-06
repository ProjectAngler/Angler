#!/bin/bash
# Jenny 2.0 healer: keeps her API alive whenever her model is serving, and
# lets a pending memory rebuild run once she is idle. Every two minutes.
COUNT=/run/jenny2-heal.count
ST=/opt/angler/state/project-angler/jenny2-interactive-qwen38-v1
model_ok() { curl -fsS --max-time 3 http://127.0.0.1:30000/v1/models >/dev/null 2>&1; }
api_ok()   { curl -fsS --max-time 5 http://127.0.0.1:8088/health >/dev/null 2>&1; }
api_json() { curl -s --max-time 5 http://127.0.0.1:8088/health 2>/dev/null; }
api_idle() { curl -s --max-time 5 http://127.0.0.1:8088/v1/ui-state 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); a=d.get('activity') or {}; ph=str(a.get('phase') or 'IDLE').upper(); print('yes' if ph in ('IDLE','ERROR','QUIESCENT','WAITING','COMMITTED','DONE') else 'no')" 2>/dev/null; }
recent_chat() { journalctl -u jenny2-api --since "5 min ago" --no-pager 2>/dev/null | grep -c "POST /v1/chat"; }

if ! model_ok; then
  echo "model not serving; nothing to heal (docker restart policy owns the model)"; echo 0 > "$COUNT"; exit 0
fi
state=$(systemctl is-active jenny2-api.service)
if [ "$state" != "active" ]; then
  echo "api is $state while model serves: reset-failed + start"
  systemctl reset-failed jenny2-api.service; systemctl start jenny2-api.service; echo 0 > "$COUNT"; exit 0
fi
if api_ok; then
  echo 0 > "$COUNT"
  # a memory rebuild is requested (projection fault) and she is idle: restart so it runs
  if [ -f "$ST/memory-rebuild-requested.json" ] && [ "$(recent_chat)" = "0" ] && [ "$(api_idle)" = "yes" ]; then
    echo "memory rebuild requested and she is idle: restarting to rebuild from canonical"
    systemctl restart jenny2-api.service
  fi
  exit 0
fi
# health answers but not OK (degraded): leave her be unless a rebuild is pending and she is idle
if api_json | grep -q '"api_ready":true'; then
  if [ -f "$ST/memory-rebuild-requested.json" ] && [ "$(recent_chat)" = "0" ] && [ "$(api_idle)" = "yes" ]; then
    echo "degraded with a rebuild pending and idle: restarting to rebuild"; systemctl restart jenny2-api.service
  else
    echo "degraded but answering; not restarting"
  fi
  echo 0 > "$COUNT"; exit 0
fi
n=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$COUNT"
if [ "$n" -ge 2 ]; then echo "api active but not answering health ($n checks): restart"; systemctl restart jenny2-api.service; echo 0 > "$COUNT"
else echo "api not answering health (check $n of 2); waiting one more interval"; fi
