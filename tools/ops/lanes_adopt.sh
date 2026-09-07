#!/bin/bash
# Adopt the three-lane server: same as jenny-qwen38-dflash3-p2p plus
# --max-running-requests 3, cuda-graph decode bs 3, mamba cache 9.
# Pool drops 77.6K -> 37.4K tokens; stage budgets are 8-12K each.
set -e
OLD=jenny-qwen38-dflash3-p2p; NEW=jenny-qwen38-lanes3
sudo docker rm -f $NEW >/dev/null 2>&1 || true
IMAGE=$(sudo docker inspect $OLD --format '{{.Config.Image}}')
ENVARGS=(); for name in CUDA_VISIBLE_DEVICES SGLANG_DISABLE_SILU_FP4_QUANT_FUSION TRANSFORMERS_OFFLINE HF_HUB_OFFLINE; do val=$(sudo docker inspect $OLD --format '{{range .Config.Env}}{{println .}}{{end}}' | grep "^$name=" | head -1 | cut -d= -f2-); [ -n "$val" ] && ENVARGS+=(-e "$name=$val"); done
MOUNTARGS=(); while IFS='|' read -r src dst mode; do [ -z "$src" ] && continue; if [ "$mode" = "ro" ]; then MOUNTARGS+=(-v "$src:$dst:ro"); else MOUNTARGS+=(-v "$src:$dst"); fi; done < <(sudo docker inspect $OLD --format '{{range .Mounts}}{{.Source}}|{{.Destination}}|{{.Mode}}{{println}}{{end}}')
CMD=$(sudo docker inspect $OLD --format '{{json .Config.Cmd}}')
NEWCMD=$(python3 -c "
import json,sys; c=json.loads(sys.argv[1])
def setflag(f,v):
    i=c.index(f); c[i+1]=v
setflag('--max-running-requests','3'); setflag('--cuda-graph-max-bs-decode','3'); setflag('--cuda-graph-bs-decode','3'); setflag('--max-mamba-cache-size','9')
print('\n'.join(c))" "$CMD")
mapfile -t CMDARR <<< "$NEWCMD"
sudo systemctl stop jenny2-heal.timer jenny2-her-job.service jenny2-api.service
sudo docker stop -t 30 $OLD >/dev/null; sudo docker update --restart=no $OLD >/dev/null
sudo docker run -d --name $NEW --restart=always --gpus all --ipc=host --network=host --shm-size=64m "${ENVARGS[@]}" "${MOUNTARGS[@]}" "$IMAGE" "${CMDARR[@]}" >/dev/null
for i in $(seq 1 120); do curl -fsS --max-time 2 http://127.0.0.1:30000/v1/models >/dev/null 2>&1 && { echo "lanes server serving after ~$((i*5))s"; break; }; sleep 5; done
sudo docker logs $NEW 2>&1 | grep -E "max_total_num_tokens" | tail -1 | cut -c1-160
sudo systemctl start jenny2-api.service; for i in $(seq 1 100); do curl -fsS --max-time 3 http://127.0.0.1:8088/health >/dev/null 2>&1 && { echo "api healthy after ~$((i*2))s"; break; }; sleep 2; done
sudo systemctl start jenny2-her-job.service jenny2-heal.timer
echo "ADOPTED $NEW (rollback: docker stop $NEW; docker update --restart=always $OLD; docker start $OLD)"
