#!/usr/bin/env bash
# 只跑 analyze_v2 + 打包,适用于:
#   - 训练已经跑完,JSON 都在 /workspace/runs/<tag>/results/ 下
#   - 但收尾打包失败(例如 unbound variable bug)需要补救
#
# 用法:
#   bash scripts/pack_results.sh configs/v2_baseline.yaml
#   bash scripts/pack_results.sh configs/v2_sync.yaml
#   bash scripts/pack_results.sh configs/v2.yaml

set -eo pipefail   # 注意:不加 -u,避免再被 unbound variable 坑

CONFIG="${1:-configs/v2.yaml}"

cd /workspace/pp-preempt-v2
export PATH="/opt/conda/bin:${PATH:-}"

# 确保 pyyaml
python3 -c "import yaml" 2>/dev/null || python3 -m pip install --quiet pyyaml>=6.0

# 从 config 读 results_dir
eval "$(python3 - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print("RESULTS_DIR=" + c["output"]["results_dir"])
PY
)"

CFG_TAG=$(basename "${CONFIG}" .yaml)
STAMP=$(date +%Y%m%d_%H%M%S)
DOWNLOAD_ROOT="/workspace/downloads"
BUNDLE_NAME="async-1f1b_${CFG_TAG}_${STAMP}"
BUNDLE_DIR="${DOWNLOAD_ROOT}/${BUNDLE_NAME}"
mkdir -p "${BUNDLE_DIR}"

echo "[pack] CONFIG      = ${CONFIG}"
echo "[pack] RESULTS_DIR = ${RESULTS_DIR}"
echo "[pack] BUNDLE_DIR  = ${BUNDLE_DIR}"

# 数下有几个 JSON
N_JSON=$(ls "${RESULTS_DIR}"/injection_k*_trial*.json 2>/dev/null | wc -l)
echo "[pack] found ${N_JSON} trial JSON files"

# 1) 跑分析,把表存成文本
python3 analyze_v2.py --config "${CONFIG}" \
  > "${BUNDLE_DIR}/summary.txt" 2>&1 || true

# 2) 拷 JSON
cp "${RESULTS_DIR}"/injection_k*_trial*.json "${BUNDLE_DIR}/" 2>/dev/null || true

# 3) 拷 CSV
cp "${RESULTS_DIR}"/summary_v2.csv "${BUNDLE_DIR}/" 2>/dev/null || true

# 4) 拷 config
cp "${CONFIG}" "${BUNDLE_DIR}/config_used.yaml"

# 5) 写说明
cat > "${BUNDLE_DIR}/README.txt" <<EOF
pp-preempt-v2-async-1f1b 实验结果(补救打包)
============================================
完成时间    : ${STAMP}
config      : ${CONFIG}
trial 数    : ${N_JSON}
说明        : 由 scripts/pack_results.sh 单独打包,
              而非 run_all_v2.sh 末尾自动打包。
EOF

# 6) tar.gz
( cd "${DOWNLOAD_ROOT}" && tar czf "${BUNDLE_NAME}.tar.gz" "${BUNDLE_NAME}" )

echo ""
echo "============================================================"
echo " 打包完成:"
echo "   目录   : ${BUNDLE_DIR}/"
echo "   tar.gz : ${DOWNLOAD_ROOT}/${BUNDLE_NAME}.tar.gz"
echo ""
ls -la "${BUNDLE_DIR}" | sed 's/^/   /'
echo ""
ls -la "${DOWNLOAD_ROOT}/${BUNDLE_NAME}.tar.gz" | sed 's/^/   /'
echo "============================================================"
