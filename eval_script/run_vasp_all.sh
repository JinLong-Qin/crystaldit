#!/usr/bin/env bash
set -euo pipefail

# ...existing code...
# 调整为实际并行进程数（可由第二个参数覆盖）
NP=${2:-32}

# 根目录由第一个参数手动传入（默认 ./chgnet_results）
ROOT=${1:-./chgnet_results}

# 需要运行的基目录列表（保持 dft_un / dft_non_un 不变）
BASE_DIRS=("$ROOT/dft_un" "$ROOT/dft_non_un")

# 可选：如果需要加载模块（取消注释并按需修改）
# module purge
# module load intel/19.0.5 mpi/openmpi/4.0.3 vasp/6.3.0

for base in "${BASE_DIRS[@]}"; do
  [ -d "$base" ] || { echo "目录不存在: $base"; continue; }
  # 遍历 base 下的直接子目录（每个子目录为一个计算）
  for d in "$base"/*; do
    [ -d "$d" ] || continue
    echo "=== 进入 $d ==="
    cd "$d"
    # 简单跳过已经有 OUTCAR 的目录（如需更严格检测可改写）
    if [ -f OUTCAR ]; then
      echo "跳过：已存在 OUTCAR"
      cd - >/dev/null
      continue
    fi
    # 运行 VASP
    echo "运行: mpirun -np $NP vasp_std > vasp.log 2>&1"
    mpirun -np $NP vasp_std > vasp.log 2>&1 || {
      echo "VASP 运行失败：$d (查看 vasp.log)"
      cd - >/dev/null
      continue
    }
    echo "完成: $d"
    cd - >/dev/null
  done
done

echo "全部完成"
# ...existing code...