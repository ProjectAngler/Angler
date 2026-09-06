#!/bin/bash
# Rollback the P2P switch: remove the patched-module overlay (stock 610.43.02
# modules from Ubuntu's package resume), restore GRUB, rebuild initramfs,
# and re-enable the model container's autostart. Then reboot.
# If the 610 userspace itself is a problem afterwards:
#   sudo apt-get install -y nvidia-driver-595-open
set -e
KVER=$(uname -r)
sudo rm -rf "/lib/modules/$KVER/updates/nvidia-p2p"
sudo depmod -a "$KVER"
echo -n "nvidia now resolves to: "; modinfo -F filename nvidia -k "$KVER"
if [ -f /etc/default/grub.bak-p2p ]; then
  sudo cp /etc/default/grub.bak-p2p /etc/default/grub
else
  sudo sed -i 's/ amd_iommu=on iommu=pt//' /etc/default/grub
fi
grep -E "^GRUB_CMDLINE_LINUX_DEFAULT" /etc/default/grub
sudo update-grub 2>&1 | tail -1
sudo update-initramfs -u 2>&1 | tail -1
sudo docker update --restart=always jenny-qwen38-dflash3 >/dev/null 2>&1 && echo "model container autostart: on"
echo "ROLLBACK STAGED. Now run:   sudo reboot"
