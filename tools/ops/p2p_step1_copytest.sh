#!/bin/bash
# Step 1 of the safe P2P investigation: the smallest possible peer copy.
# Her services are stopped first so nothing is mid-write if the box freezes.
# If the machine hangs here, the P2P patch is unusable on this hardware:
# power-cycle, then run p2p_rollback.sh and reboot.
D=/home/becca/jenny-p2p
sudo systemctl stop jenny2-her-job.service jenny2-api.service
sudo docker stop -t 20 jenny-qwen38-dflash3 >/dev/null 2>&1
sync
echo "services stopped, disks synced. Peer copy test in 5 seconds; be ready to power-cycle."
sleep 5
cd /opt/angler/src/angler
CUDA_VISIBLE_DEVICES=0,1 timeout 120 /opt/angler/venvs/angler/bin/python "$D/p2p_test.py" 2>&1 | tail -5 | tee "$D/p2p_after_step1.txt"
echo "=== kernel log during the test"
sudo journalctl -k --since "2 min ago" --no-pager | grep -iE "NVRM|Xid|AMD-Vi|fault|lockup" | tail -8
echo "STEP 1 DONE (box survived). Before: 19.9 / 20.6 GB/s, peer access False."
