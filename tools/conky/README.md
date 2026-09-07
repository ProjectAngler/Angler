# Angler workstation monitor (conky)

Becca's desktop overlay for the Jenny/Angler workstation: GPU utilization and
memory history for both cards, drawn by `gpu_graphs.lua`, configured by
`angler.conf`, autostarted by `angler-monitor.desktop`.

Install: copy `angler.conf` and `gpu_graphs.lua` to `~/.config/conky/`, and
`angler-monitor.desktop` to `~/.config/autostart/`. Requires `conky` with Lua
support and `nvidia-smi`.
