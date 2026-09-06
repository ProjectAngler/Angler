#!/bin/bash
# Step 2: her model server with its custom all-reduce disabled, so NCCL
# carries the tensor-parallel traffic over the now-working P2P path.
# Same image, mounts, flags, and env as jenny-qwen38-dflash3, plus one flag.
# The new container is jenny-qwen38-dflash3-p2p, autostart OFF until proven.
# If the box freezes: power-cycle, run p2p_rollback.sh, reboot.
set -e
OLD=jenny-qwen38-dflash3
NEW=jenny-qwen38-dflash3-p2p
D=/home/becca/jenny-p2p

sudo systemctl stop jenny2-her-job.service jenny2-api.service 2>/dev/null || true
sudo docker stop -t 20 "$OLD" >/dev/null 2>&1 || true
sudo docker rm -f "$NEW" >/dev/null 2>&1 || true

IMAGE=$(sudo docker inspect "$OLD" --format '{{.Config.Image}}')
ENVARGS=()
for name in CUDA_VISIBLE_DEVICES SGLANG_DISABLE_SILU_FP4_QUANT_FUSION TRANSFORMERS_OFFLINE HF_HUB_OFFLINE; do
  val=$(sudo docker inspect "$OLD" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep "^$name=" | head -1 | cut -d= -f2-)
  [ -n "$val" ] && ENVARGS+=(-e "$name=$val")
done
MOUNTARGS=()
while IFS='|' read -r src dst mode; do
  [ -z "$src" ] && continue
  if [ "$mode" = "ro" ]; then MOUNTARGS+=(-v "$src:$dst:ro"); else MOUNTARGS+=(-v "$src:$dst"); fi
done < <(sudo docker inspect "$OLD" --format '{{range .Mounts}}{{.Source}}|{{.Destination}}|{{.Mode}}{{println}}{{end}}')
# the original command, plus the one flag
CMD=$(sudo docker inspect "$OLD" --format '{{json .Config.Cmd}}')
NEWCMD=$(python3 -c "import json,sys; c=json.loads(sys.argv[1]); c.append('--disable-custom-all-reduce'); print('\n'.join(c))" "$CMD")
mapfile -t CMDARR <<< "$NEWCMD"

echo "== starting $NEW (custom all-reduce disabled; NCCL over P2P)"
sync
sudo docker run -d --name "$NEW" --restart=no --gpus all --ipc=host --network=host --shm-size=64m \
  "${ENVARGS[@]}" "${MOUNTARGS[@]}" "$IMAGE" "${CMDARR[@]}" >/dev/null
echo "container started; waiting for the model (up to 12 minutes). Be ready to power-cycle."
for i in $(seq 1 144); do
  if curl -fsS --max-time 2 http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then echo "MODEL SERVING after ~$((i*5))s"; break; fi
  if [ "$(sudo docker inspect -f '{{.State.Status}}' "$NEW")" != "running" ]; then echo "container exited; last log lines:"; sudo docker logs --tail 30 "$NEW" 2>&1 | tail -30; exit 1; fi
  sleep 5
done
curl -fsS --max-time 2 http://127.0.0.1:30000/v1/models >/dev/null 2>&1 || { echo "not serving after 12 min; logs:"; sudo docker logs --tail 30 "$NEW" 2>&1 | tail -30; exit 1; }
echo "== kernel log during load"
sudo journalctl -k --since "15 min ago" --no-pager | grep -iE "NVRM|Xid|AMD-Vi|fault|lockup" | tail -6
echo "== decode speed (before: ~96 tok/s)"
/opt/angler/venvs/angler/bin/python "$D/decode_bench.py" | tee "$D/p2p_after_step2.txt"
echo "STEP 2 DONE. If speed is good and the box is stable, run p2p_adopt.sh"
