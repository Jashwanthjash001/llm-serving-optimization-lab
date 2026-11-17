#!/usr/bin/env bash
#
# audit_gpu_server.sh — Stage 0 inventory of the Ubuntu GPU serving box.
#
# READ-ONLY. Installs nothing, runs no inference, downloads nothing, changes no
# config. Run it on the Ubuntu server and paste the output back.
#
#   bash audit_gpu_server.sh            # prints to screen
#   bash audit_gpu_server.sh > audit.txt
#
set -u
line(){ printf '\n===== %s =====\n' "$1"; }

line "OS / kernel"
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
uname -a

line "CPU"
lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Socket|Thread' || grep -m1 'model name' /proc/cpuinfo

line "RAM"
free -h 2>/dev/null || grep MemTotal /proc/meminfo

line "Disk space"
df -h / /home 2>/dev/null | awk 'NR==1 || /\/($|home)/'

line "NVIDIA GPUs (model + VRAM)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap --format=csv
  echo "--- topology (NVLink / PCIe between the two GPUs) ---"
  nvidia-smi topo -m 2>/dev/null | head -12
else
  echo "nvidia-smi NOT found — driver may not be installed."
fi

line "CUDA toolkit"
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2 || echo "nvcc not found (runtime CUDA may still exist via driver)"

line "Docker + NVIDIA Container Toolkit"
command -v docker >/dev/null 2>&1 && docker --version || echo "docker not found"
command -v nvidia-ctk >/dev/null 2>&1 && nvidia-ctk --version | head -1 || echo "nvidia-ctk (container toolkit) not found"
docker images 2>/dev/null | grep -iE 'vllm|cuda|pytorch|tgi|llama' | head || echo "(no relevant docker images, or docker needs sudo)"

line "Running containers / inference services"
docker ps 2>/dev/null | head || echo "(docker ps needs permission, or none running)"
ss -ltnp 2>/dev/null | grep -E ':(8000|8080|8001|11434)' || echo "(no server on common inference ports 8000/8080/11434)"

line "vLLM / serving Python packages"
for py in python3 python; do command -v $py >/dev/null 2>&1 && { echo "[$py] $($py --version 2>&1)"; $py -m pip show vllm 2>/dev/null | grep -E 'Name|Version' || echo "  vllm: not installed in $py"; break; }; done

line "Existing model caches / downloaded models"
du -sh ~/.cache/huggingface 2>/dev/null || echo "no HF cache"
ls -1 ~/.cache/huggingface/hub 2>/dev/null | head || true
find ~ -maxdepth 3 -iname '*.gguf' -o -iname '*.safetensors' 2>/dev/null | head

line "DONE"
echo "Paste everything above back into the chat."
