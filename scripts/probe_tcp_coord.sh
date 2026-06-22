#!/usr/bin/env bash
# 4-worker 验证脚本: 测试 TCPCoordinator 能不能跨 pod 通信。
#
# 用法(平台启动命令,4 worker):
#   cd /workspace/pp-preempt-v2 && bash scripts/probe_tcp_coord.sh
#
# 预期输出:
#   每个 rank 打印 "✓ TCP coord OK ! rank X heard from rank Y"
#   如果某个 rank 卡在 signal/wait,说明 pod 间网络不通
#   (这跟共享存储无关,TCP 走 NCCL 同一条网络)

set -eo pipefail

cd /workspace/pp-preempt-v2

# 解析 rank (从平台环境变量)
RANK="${VC_TASK_INDEX:-0}"
MASTER_ADDR_RAW="${VC_WORKER_HOSTS_0:-}"

# 解析 master IP (复用 run_all_v2.sh 的 fallback)
if [ -z "$MASTER_ADDR_RAW" ]; then
  echo "ERROR: VC_WORKER_HOSTS_0 未设置" >&2
  exit 1
fi

MASTER_IP=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_ADDR_RAW}'))" 2>/dev/null || echo "$MASTER_ADDR_RAW")
echo "[probe rank=$RANK] master=$MASTER_IP"

# 等一下,让 rank 0 先起来
[ "$RANK" != "0" ] && sleep 3

${PY:-python3} - <<PY
import sys, time
sys.path.insert(0, '/workspace/pp-preempt-v2')
from coordinator import TCPCoordinator

RANK = $RANK
W = 4
PORT = 29555  # probe 用,与正式实验端口错开

print(f"[rank {RANK}] connecting TCPStore master=$MASTER_IP:{PORT} ...", flush=True)
coord = TCPCoordinator(
    master_addr="$MASTER_IP", master_port=PORT,
    rank=RANK, world_size=W,
)
print(f"[rank {RANK}] ✓ TCPStore connected", flush=True)

# 测试 1: signal/wait
if RANK == 0:
    time.sleep(2)  # 让 client 先去 wait
    coord.signal("hello_from_0", "world")
    print(f"[rank 0] sent signal", flush=True)
else:
    v = coord.wait_signal("hello_from_0", timeout=30)
    print(f"[rank {RANK}] ✓ heard from rank 0: {v}", flush=True)

# 测试 2: barrier
print(f"[rank {RANK}] entering barrier ...", flush=True)
wait_s = coord.barrier("all_done", RANK, timeout=30)
print(f"[rank {RANK}] ✓ barrier passed (waited {wait_s:.2f}s)", flush=True)

print(f"[rank {RANK}] ✓✓✓ TCP coordinator works", flush=True)
PY
