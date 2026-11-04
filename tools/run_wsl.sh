#!/usr/bin/env bash
set -euo pipefail
# Run as project user aienge; keep Python/CUDA libraries on Linux filesystem.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${JAMSIL_GPU_VENV:-$HOME/.venvs/jamsil-tf}"
if [[ ! -x "$venv_dir/bin/python" ]]; then
  echo "Missing GPU venv: $venv_dir" >&2
  exit 1
fi
nvidia_dir="$venv_dir/lib/python3.12/site-packages/nvidia"
export PATH="$nvidia_dir/cuda_nvcc/bin:$venv_dir/bin:$PATH"
for library_dir in "$nvidia_dir"/*/lib; do
  if [[ -d "$library_dir" ]]; then
    export LD_LIBRARY_PATH="$library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
cd -- "$repo_dir"
exec "$venv_dir/bin/python" "$@"
