#!/bin/bash
# Adopt the P2P container: it becomes the one that starts at boot; the old
# one stays stopped as rollback. Then her services come back up.
set -e
sudo docker update --restart=always jenny-qwen38-dflash3-p2p >/dev/null
sudo docker update --restart=no jenny-qwen38-dflash3 >/dev/null
sudo systemctl start jenny2-api.service
for i in $(seq 1 100); do curl -fsS --max-time 3 http://127.0.0.1:8088/health >/dev/null 2>&1 && { echo "api healthy after ~$((i*2))s"; break; }; sleep 2; done
sudo systemctl start jenny2-her-job.service
systemctl is-active jenny2-api.service jenny-console-proxy.service jenny2-her-job.service | tr '\n' ' '; echo
echo "ADOPTED: jenny-qwen38-dflash3-p2p is her model server; jenny-qwen38-dflash3 kept stopped for rollback."
