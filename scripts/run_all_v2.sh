#!/usr/bin/env bash
# PP 抢占恢复代价 v2 — 多节点启动脚本
# 平台镜像：pytorch:2.9.1-cuda12.6-cudnn9-py311-ubuntu22.04
# 平台：1 worker × 4 replicas
#
# 平台注入的环境变量：
#   VC_WORKER_HOSTS_0  — worker-0 的主机名（可能 DNS 解析失败，脚本会自动绕过）
#   VC_TASK_INDEX      — 当前节点序号（0, 1, 2, 3）
#
# 4 个节点执行完全相同的命令：
#   cd /workspace
#   bash scripts/run_all_v2.sh \
#     --node-rank  ${VC_TASK_INDEX} \
#     --master-addr ${VC_WORKER_HOSTS_0} \
#     [--install-deps]

set -euo pipefail

CONFIG="${CONFIG:-configs/v2.yaml}"
NNODES="${NNODES:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-}}"
# VC_WORKER_HOSTS_0 可能是 hostname（DNS 可能失败），下面会自动解析为 IP
MASTER_ADDR_RAW="${MASTER_ADDR:-${VC_WORKER_HOSTS_0:-}}"
STAGES="${STAGES:-1 2 3 4}"
REPEATS="${REPEATS:-3}"
RDZV_PREFIX="${RDZV_PREFIX:-pp_preempt_v2}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
CODE_DIR="${CODE_DIR:-/workspace/pp-preempt-v2}"

# 共享存储上的 master IP 文件路径在解析 config 后由 COORD_DIR 派生(per-config namespace),
# 见后面的 "从 config 解析所有路径" 段。

usage() {
  cat <<'USAGE'
用法：
  scripts/run_all_v2.sh --node-rank RANK --master-addr HOST [选项]

必填：
  --node-rank N      当前节点 rank（= VC_TASK_INDEX，0-3）
  --master-addr HOST rank 0 的主机名或 IP（= VC_WORKER_HOSTS_0）

选项：
  --config PATH      配置文件，默认 configs/v2.yaml
  --nnodes N         总节点数，默认 4
  --master-port PORT 训练 rendezvous 端口，默认 29500
  --stages "1 2 3 4" 要测试的 k 列表
  --repeats N        每个 k 重复次数，默认 3
  --install-deps     运行前先 pip install -r requirements.txt
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-rank)    NODE_RANK="$2";        shift 2 ;;
    --master-addr)
      [[ -z "${2:-}" || "${2}" == --* ]] && {
        echo "ERROR: --master-addr 需要一个值" >&2; exit 2; }
      MASTER_ADDR_RAW="$2"; shift 2 ;;
    --config)       CONFIG="$2";           shift 2 ;;
    --nnodes)       NNODES="$2";           shift 2 ;;
    --master-port)  MASTER_PORT="$2";      shift 2 ;;
    --stages)       STAGES="$2";           shift 2 ;;
    --repeats)      REPEATS="$2";          shift 2 ;;
    --install-deps) INSTALL_DEPS="1";      shift   ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -z "${NODE_RANK}" ]] && {
  echo "ERROR: 未设置 NODE_RANK（传 --node-rank 或设置 VC_TASK_INDEX）" >&2; exit 2; }
[[ -z "${MASTER_ADDR_RAW}" ]] && {
  echo "ERROR: 未设置 MASTER_ADDR（传 --master-addr 或设置 VC_WORKER_HOSTS_0）" >&2; exit 2; }

cd "${CODE_DIR}"
export PATH="/opt/conda/bin:${PATH:-}"

# 路径解析需要 pyyaml; 提前安装(原 install-deps 在路径解析之后,会迟到)
if ! ${PY:-python3} -c "import yaml" 2>/dev/null; then
  ${PY:-python3} -m pip install --quiet pyyaml>=6.0 2>/dev/null || true
fi

# ── 从 config 解析所有路径(三档并发跑不互相覆盖的关键)─────────────────────
#
# 三个 config 把 ckpts/results/coord/logs 分别放到
# /workspace/runs/{baseline,sync,async}/...
# 下面用 python 解析 YAML 把这些路径读出来,本脚本不再写死路径。
parse_cfg_paths() {
  ${PY:-python3} - "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print("CKPT_DIR=" + cfg["checkpoint"]["dir"])
print("COORD_DIR=" + cfg["injection"]["coord_dir"])
print("RESULTS_DIR=" + cfg["output"]["results_dir"])
print("LOGS_DIR=" + cfg["output"]["logs_dir"])
PY
}
eval "$(parse_cfg_paths "${CONFIG}")"

# CFG_TAG 用于:1) 下载包名 prefix,2) sanity 日志
# (实际的 per-config 隔离已经在 YAML 里通过 runs/{baseline,sync,async}/ 路径实现)
CFG_TAG=$(basename "${CONFIG}" .yaml)

mkdir -p "${CKPT_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}" "${COORD_DIR}"
DOWNLOAD_ROOT="/workspace/downloads"
mkdir -p "${DOWNLOAD_ROOT}"

# master_ip.txt 也按 RUN_TAG 分,这样三档并发跑时 rank 0 互不覆盖
MASTER_IP_FILE="${COORD_DIR}/master_ip.txt"

echo "[paths] CFG_TAG     = ${CFG_TAG}"
echo "[paths] CKPT_DIR    = ${CKPT_DIR}"
echo "[paths] RESULTS_DIR = ${RESULTS_DIR}"
echo "[paths] COORD_DIR   = ${COORD_DIR}"
echo "[paths] MASTER_IP   = ${MASTER_IP_FILE}"

# ── 关键:防止上一次 job 残留的 master_ip.txt 把 worker 引到死 IP ─────────
#
# 此前的 bug:master_ip.txt 是共享存储里的持久文件,上次 job 写入的旧 IP
# 不会自动清理。这次 job 启动时,worker 2/3 比 worker 0 先到达,直接读到
# 残留的旧 IP,跟一个不存在的 pod 做 rendezvous → 10 分钟 timeout 崩溃。
#
# 修法:
#   - 记录本脚本启动时间戳 (SCRIPT_START_EPOCH)
#   - rank 0 在写新 IP 之前先 unlink 旧文件
#   - 其他 rank 等待时,不仅要文件存在,还要 mtime ≥ SCRIPT_START_EPOCH,
#     这样即使旧 IP 文件还在,也会被忽略,直到 rank 0 重写后才接受
SCRIPT_START_EPOCH=$(date +%s)

# ── 关键：用 IP 替换 hostname，彻底绕开 DNS 解析失败 ─────────────────────────
#
# VC_WORKER_HOSTS_0 是 worker-0 的 hostname，容器内 DNS 可能解析失败。
# 解决方案：
#   rank 0 → 用 `hostname -i` 获取自己的真实 IPv4，写到共享文件
#   其他 rank → 轮询等待文件出现且 mtime 是本次 job 的，读取 IP
#
resolve_master_ip() {
  # 关键：函数内所有打印信息必须走 stderr（>&2），
  # 只有最后一行纯 IP 走 stdout，才能被 $(...) 正确捕获。
  if [[ "${NODE_RANK}" == "0" ]]; then
    # 先删旧文件(避免 rank 1/2/3 读到上次 job 的残留 IP)
    rm -f "${MASTER_IP_FILE}"
    echo "[rank 0] 已清理旧的 master_ip.txt(如有)" >&2

    MY_IP=$(python3 -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('10.255.255.255', 1))
    ip = s.getsockname()[0]
    s.close()
    if not ip.startswith('127.'):
        print(ip); exit(0)
except Exception:
    pass
import subprocess
out = subprocess.check_output(['hostname', '-I'], text=True).strip().split()
for ip in out:
    if not ip.startswith('127.'):
        print(ip); exit(0)
print(socket.gethostbyname(socket.gethostname()))
" 2>/dev/null || hostname -i | awk '{print $1}')

    echo "[rank 0] 本机 IP = ${MY_IP}" >&2          # ← 打印走 stderr
    # 原子写:先写临时文件,再 mv,避免其他 rank 读到半行
    echo "${MY_IP}" > "${MASTER_IP_FILE}.tmp"
    mv -f "${MASTER_IP_FILE}.tmp" "${MASTER_IP_FILE}"
    echo "${MY_IP}"                                   # ← 只有这行走 stdout

  else
    echo "[rank ${NODE_RANK}] 等待 master IP 文件(必须是本次 job 写入)..." >&2
    waited=0
    while true; do
      if [[ -f "${MASTER_IP_FILE}" ]]; then
        # 只接受脚本启动后才创建/更新的文件,忽略上一次 job 的残留
        file_mtime=$(stat -c %Y "${MASTER_IP_FILE}" 2>/dev/null || echo 0)
        if [[ "${file_mtime}" -ge "${SCRIPT_START_EPOCH}" ]]; then
          break
        else
          # 旧文件,继续等
          if [[ $((waited % 10)) -eq 0 ]]; then
            echo "[rank ${NODE_RANK}] master_ip.txt 是旧文件(mtime=${file_mtime} < start=${SCRIPT_START_EPOCH}),等待 rank 0 重写..." >&2
          fi
        fi
      fi
      sleep 1
      waited=$((waited + 1))
      if [[ $waited -ge 120 ]]; then
        echo "ERROR: 等待 master IP 超时（120s）" >&2
        python3 -c "import socket; print(socket.gethostbyname('${MASTER_ADDR_RAW}'))" \
          2>/dev/null || echo "${MASTER_ADDR_RAW}"
        return
      fi
    done
    ip=$(cat "${MASTER_IP_FILE}")
    echo "[rank ${NODE_RANK}] 获取到 master IP = ${ip}" >&2  # ← stderr
    echo "${ip}"                                               # ← 只有这行走 stdout
  fi
}

# 如果环境变量 FORCE_MASTER_IP 已显式给出(SSH 直连场景),跳过 resolve_master_ip 的
# master_ip.txt 共享文件流程,直接用,这样 4 个独立容器(/workspace 不共享)也能跑。
if [[ -n "${FORCE_MASTER_IP:-}" ]]; then
  MASTER_ADDR="${FORCE_MASTER_IP}"
  echo "[paths] FORCE_MASTER_IP set, skipping master_ip.txt resolution"
else
  MASTER_ADDR=$(resolve_master_ip)
fi
echo "MASTER_ADDR = ${MASTER_ADDR}:${MASTER_PORT}"

# ── 环境变量（PyTorch 2.9.1 + CUDA 12.6）────────────────────────────────────
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-60}"

# 强制 IPv4，避免 IPv6 解析失败的警告和连接问题
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_FAMILY="${GLOO_SOCKET_FAMILY:-AF_INET}"

# 自动检测网络接口（通常是 eth0）
IFACE=$(ip route 2>/dev/null | awk '/default/{print $5; exit}')
IFACE="${IFACE:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${IFACE}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${IFACE}}"

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

[[ "${INSTALL_DEPS}" == "1" ]] && {
  echo "=== Installing deps (pyyaml only, no torch download) ==="
  # 只安装 pyyaml；torch 已预装，不走网络
  python -m pip install pyyaml>=6.0 2>/dev/null || true
}

echo "========================================"
echo " PP preempt v2 (fully self-contained)"
echo "  torch image  = pytorch:2.9.1-cuda12.6"
echo "  node_rank    = ${NODE_RANK} / ${NNODES}"
echo "  master_ip    = ${MASTER_ADDR}:${MASTER_PORT}"
echo "  net_iface    = ${IFACE}"
echo "  config       = ${CONFIG}"
echo "  stages       = ${STAGES}"
echo "  repeats      = ${REPEATS}"
echo "========================================"

# rank 0 在 trial 循环开始前清掉上一次 run 的 results JSON,
# 否则 analyze_v2 会把上次的 sync/async 结果混进新表。
# 只清 JSON,不删 ckpt(ckpt 复用可省 warmup 时间? — 不行,会跨 config 错位,也清)。
if [[ "${NODE_RANK}" == "0" ]]; then
  rm -f "${RESULTS_DIR}"/injection_k*_trial*.json \
        "${RESULTS_DIR}"/summary_v2.csv \
        2>/dev/null || true
  rm -rf "${CKPT_DIR}"/*.pt 2>/dev/null || true
  echo "[rank 0] cleared old results + ckpts in ${RESULTS_DIR} / ${CKPT_DIR}"
fi

for k in ${STAGES}; do
  for trial in $(seq 0 $((REPEATS - 1))); do
    rdzv_id="${RDZV_PREFIX}_k${k}_t${trial}"
    echo ""
    echo "=== k=${k} trial=${trial} node_rank=${NODE_RANK} ==="

    # rank 0 清理上一个 trial 的协调文件（保留 master_ip.txt）
    if [[ "${NODE_RANK}" == "0" ]]; then
      rm -f "${COORD_DIR}"/bar_* \
            "${COORD_DIR}"/sig_* 2>/dev/null || true
    fi
    sleep 2

    torchrun \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${NODE_RANK}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      --rdzv_id="${rdzv_id}" \
      run_injection_v2.py \
        --config      "${CONFIG}" \
        --k           "${k}" \
        --trial       "${trial}" \
        --master-addr "${MASTER_ADDR}" \
        --master-port "${MASTER_PORT}"

    echo "=== k=${k} trial=${trial} done ==="
    sleep 3
  done
done

echo ""
echo "=== 全部完成 (node_rank=${NODE_RANK}) ==="

# ── 仅 rank 0:打包所有结果,方便从平台下载 ────────────────────────────────
# 平台只支持提交分布式训练任务,需要把结果集中到一个固定位置(共享存储)。
# rank 0 负责:
#   1) 跑 analyze_v2.py 把汇总表 + per-trial 表写成纯文本
#   2) 把 results/ 全部 JSON + 汇总表 + 用过的 config + 简短 README 复制到
#      一个时间戳目录,并 tar.gz 一份,二者并列放在 /workspace/downloads/
if [[ "${NODE_RANK}" == "0" ]]; then
  echo ""
  echo "=== [rank 0] 收集结果到下载目录 ==="

  STAMP=$(date +%Y%m%d_%H%M%S)
  # 包名带 config basename,避免三档(baseline/sync/async)互相覆盖下载结果
  BUNDLE_NAME="async-1f1b_${CFG_TAG}_${STAMP}"
  BUNDLE_DIR="${DOWNLOAD_ROOT}/${BUNDLE_NAME}"
  mkdir -p "${BUNDLE_DIR}"

  # 1) 跑分析,把两张表都存成文本
  python analyze_v2.py --config "${CONFIG}" \
    > "${BUNDLE_DIR}/summary.txt" 2>&1 || true

  # 2) 把所有 trial JSON 拷进来(原始数据,绝不丢)
  cp "${RESULTS_DIR}"/injection_k*_trial*.json \
     "${BUNDLE_DIR}/" 2>/dev/null || true

  # 3) 如果 analyze_v2 生成了 summary_v2.csv,也带上
  cp "${RESULTS_DIR}"/summary_v2.csv \
     "${BUNDLE_DIR}/" 2>/dev/null || true

  # 4) 复制实际用过的 config(实验可复现性)
  cp "${CONFIG}" "${BUNDLE_DIR}/config_used.yaml" 2>/dev/null || true

  # 5) 写一个简短 README 说明这是哪一组、什么时候跑的
  cat > "${BUNDLE_DIR}/README.txt" <<EOF
pp-preempt-v2-async-1f1b 实验结果
=================================
完成时间    : ${STAMP}
代码版本    : async-1f1b (与 graph-opt 同源 + 1F1B 异步恢复)
config      : ${CONFIG}
nnodes      : ${NNODES}
stages (k)  : ${STAGES}
repeats     : ${REPEATS}

文件清单
--------
summary.txt              analyze_v2.py 完整输出(汇总表 + per-trial 表)
summary_v2.csv           per-trial 数据的 CSV 版本(若 pandas 可用)
injection_k*_trial*.json 每个 trial 的原始结果
config_used.yaml         运行时用的配置(供复现)
EOF

  # 6) 打 tar.gz,平台下载更方便
  ( cd "${DOWNLOAD_ROOT}" && tar czf "${BUNDLE_NAME}.tar.gz" "${BUNDLE_NAME}" )

  echo ""
  echo "============================================================"
  echo " 结果已打包,平台共享存储位置:"
  echo "   目录    : ${BUNDLE_DIR}/"
  echo "   tar.gz : ${DOWNLOAD_ROOT}/${BUNDLE_NAME}.tar.gz"
  echo ""
  echo " 文件清单:"
  ls -la "${BUNDLE_DIR}" | sed 's/^/   /'
  echo ""
  echo " tar.gz 大小:"
  ls -la "${DOWNLOAD_ROOT}/${BUNDLE_NAME}.tar.gz" | sed 's/^/   /'
  echo "============================================================"
fi
