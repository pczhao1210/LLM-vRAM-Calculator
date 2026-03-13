# LLM 显存计算器

一个面向通用 LLM 部署规划的静态前端应用，支持两种使用方式：

- 从模型推算推荐 GPU 配置
- 从现有 GPU 倒推更合适的模型规模

首版已实现：

- 内置模型目录、量化目录与 GPU 目录
- 推理、LoRA、QLoRA 三种显存估算模式
- 上下文窗口、并发量、训练序列长度、micro batch、LoRA rank 等关键参数
- A100、H100、H200、B200、GB200、B300、GB300、NVIDIA RTX PRO 6000 Blackwell 等档位的推荐逻辑
- AWS、Azure、GCP、Oracle Cloud 的主流 GPU 实例映射
- 一个可直接运行的单页前端界面

## Run

```bash
cd app
pnpm install
pnpm dev
```

或者直接从仓库根目录启动：

```bash
./start.sh
```

## Build

```bash
cd app
pnpm build
```

或者从仓库根目录执行：

```bash
./start.sh build
```

## Deploy To Azure Static Web Apps

1. 在 Azure Portal 打开你已经创建好的 Static Web App 资源。
2. 进入 Deployment token，复制部署 token。
3. 在 GitHub 仓库里创建一个 Actions secret，名称填 `AZURE_STATIC_WEB_APPS_API_TOKEN`，值填刚才复制的 token。
4. 提交并推送 `.github/workflows/azure-static-web-apps.yml`。
5. 如果你的默认分支不是 `main`，把工作流里的 `branches: [main]` 改成你的实际默认分支。
6. 推送代码后，GitHub Actions 会自动执行：

```bash
cd app
pnpm install --frozen-lockfile
pnpm build
```

然后把 `app/dist` 部署到 Azure Static Web Apps。

项目已经包含 `app/public/staticwebapp.config.json`，用于支持单页应用的路由回退，避免刷新子路由时出现 404。

## Local One-Click Publish

如果你已经创建好 Azure Static Web Apps 资源，也拿到了 deployment token，可以直接在本地一键发布：

```bash
export AZURE_STATIC_WEB_APPS_API_TOKEN="<your-token>"
./publish.sh
```

注意：当前本地一键发布更适合 `x86_64` 环境。在 `ARM64 / aarch64` 环境下，Azure Static Web Apps CLI 下载的 `StaticSitesClient` 可能因架构不匹配而无法执行。若你在 ARM 设备或 ARM WSL 上开发，建议优先使用 GitHub Actions 自动发布。

这个脚本会自动执行：

```bash
cd app
pnpm build
pnpm dlx @azure/static-web-apps-cli@latest deploy ./dist --deployment-token "$AZURE_STATIC_WEB_APPS_API_TOKEN" --env production
```

你也可以直接使用：

```bash
./start.sh publish
```

如果要发布到非默认环境，可以额外指定：

```bash
export AZURE_STATIC_WEB_APPS_DEPLOY_ENV="production"
./publish.sh
```

## Current Structure

- src/data/catalog.ts: 内置模型、量化和 GPU 档案
- src/lib/calculator.ts: 显存估算和推荐逻辑
- src/App.tsx: 页面交互与结果展示
- src/index.css: 页面视觉样式

## Cloud GPU Mapping

- 目录里的 GPU 档案按云厂商公开实例规格建模，而不是直接照搬 NVIDIA 宣传口径。
- Blackwell 相关型号已拆分为独立条目：B200、GB200、B300、GB300 分开维护；不同云上暴露显存不同的，也分别建模。
- 当前重点覆盖的官方实例系列包括：
	Azure：ND_A100_v4、NDm_A100_v4、ND_H100_v5、ND_H200_v5、ND_GB200_v6、ND_GB300_v6、NC_RTXPRO6000BSE_v6、NVads A10 v5。
	AWS：P4d、P4de、P5、P5e / P5en、P6-B200、P6-B300、P6e-GB200、G6、G6e、G7e、G5。
	GCP：A2 Standard、A2 Ultra、A3 Mega / High / Edge、A3 Ultra、A4、A4X、A4X Max、G2、G4。
	Oracle Cloud：VM.GPU.A10、BM.GPU4.8、BM.GPU.A100-v2.8、BM.GPU.H100.8、BM.GPU.H200.8、BM.GPU.B200.8、BM.GPU.GB200.4、BM.GPU.GB300.4、BM.GPU.L40S.4。
- 目录中不再使用 superchip、UltraServer、NVL72 这类聚合形态作为可选 GPU 条目，避免把整机柜/整机域的容量误当成单 GPU 容量。

## Notes

- 该项目输出的是可解释的显存估算，不直接等同于真实吞吐、时延或框架级 benchmark。
- 页面展示的云实例映射用于解释“这个显卡档位在各家云上通常对应什么实例系列”，实际可售区域和配额仍需以云厂商控制台为准。
- 目录中的模型结构参数为工程估算用近似值，后续可以继续补充更细粒度档案。
