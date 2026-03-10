#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$ROOT_DIR/app"
MODE="${1:-dev}"

ensure_pnpm() {
  if ! command -v pnpm >/dev/null 2>&1; then
    echo "[start.sh] pnpm 未安装，尝试通过 npm 全局安装..."
    npm install -g pnpm
  fi
}

ensure_dependencies() {
  if [ ! -d "$APP_DIR/node_modules" ]; then
    echo "[start.sh] 安装前端依赖..."
    cd "$APP_DIR"
    pnpm install
  fi
}

run_dev() {
  cd "$APP_DIR"
  echo "[start.sh] 启动开发服务器: http://127.0.0.1:5173"
  pnpm dev --host 0.0.0.0 --port 5173
}

run_build() {
  cd "$APP_DIR"
  echo "[start.sh] 执行生产构建..."
  pnpm build
}

run_preview() {
  cd "$APP_DIR"
  echo "[start.sh] 启动预览服务器: http://127.0.0.1:4173"
  pnpm preview --host 0.0.0.0 --port 4173
}

run_publish() {
  local deploy_env
  local arch
  deploy_env="${AZURE_STATIC_WEB_APPS_DEPLOY_ENV:-production}"
  arch="$(uname -m)"

  if [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then
    cat <<'EOF'
[start.sh] 检测到当前机器是 ARM64 / aarch64。

Azure Static Web Apps CLI 当前下载的 StaticSitesClient 在这个环境里会因架构不匹配而卡住或执行失败。

建议改用以下任一方式发布：
  1. 使用仓库中的 GitHub Actions 工作流自动发布
  2. 在 x86_64 机器、x64 容器或云端 CI 环境中执行 ./publish.sh

当前仓库已提供自动发布工作流：
  .github/workflows/azure-static-web-apps.yml
EOF
    exit 1
  fi

  if [ -z "${AZURE_STATIC_WEB_APPS_API_TOKEN:-}" ]; then
    cat <<'EOF'
[start.sh] 缺少环境变量 AZURE_STATIC_WEB_APPS_API_TOKEN。

请先导出 Azure Static Web Apps 的 deployment token，例如：
  export AZURE_STATIC_WEB_APPS_API_TOKEN="<your-token>"

可选环境变量：
  export AZURE_STATIC_WEB_APPS_DEPLOY_ENV="production"
EOF
    exit 1
  fi

  cd "$APP_DIR"
  echo "[start.sh] 执行生产构建..."
  pnpm build

  echo "[start.sh] 发布到 Azure Static Web Apps 环境: $deploy_env"
  pnpm dlx @azure/static-web-apps-cli@latest deploy ./dist \
    --deployment-token "$AZURE_STATIC_WEB_APPS_API_TOKEN" \
    --env "$deploy_env"
}

print_help() {
  cat <<'EOF'
Usage:
  ./start.sh dev
  ./start.sh build
  ./start.sh preview
  ./start.sh publish

Default:
  ./start.sh
  等同于 ./start.sh dev

Publish:
  export AZURE_STATIC_WEB_APPS_API_TOKEN="<your-token>"
  ./start.sh publish
EOF
}

if [ ! -d "$APP_DIR" ]; then
  echo "[start.sh] 未找到 app/ 目录。"
  exit 1
fi

ensure_pnpm
ensure_dependencies

case "$MODE" in
  dev)
    run_dev
    ;;
  build)
    run_build
    ;;
  preview)
    run_preview
    ;;
  publish)
    run_publish
    ;;
  -h|--help|help)
    print_help
    ;;
  *)
    echo "[start.sh] 不支持的模式: $MODE"
    print_help
    exit 1
    ;;
esac