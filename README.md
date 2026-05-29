# tierflow-infer

TierFlow 的 GPU 推理微服务。加载 **Qwen2.5-7B** 骨干模型 + **5 个 CG-TabM 路由头** + 原型（proto）加权，对聊天消息做难度评分，供 [tierflow-core](https://github.com/NeoFii/tierflow-core) 做路由分级（tier 1-5 → 上游模型）。

- 框架：FastAPI + PyTorch + Transformers
- 端口：`8001`
- 无数据库、无 Redis、无 JWT
- 依赖：NVIDIA GPU + 本地模型权重

---

## 在 TierFlow 中的位置

TierFlow 为三节点架构，本服务部署在 **GPU 节点**：

| 节点 | 配置 | 服务 |
|------|------|------|
| 前端节点 | 2H2G | [tierflow-zh](https://github.com/NeoFii/Frontend-zh)（Next.js）+ Nginx |
| 后端节点 | 2H4G | MySQL + Redis + [tierflow-core](https://github.com/NeoFii/tierflow-core)（`app` + `worker`） |
| GPU 节点 | 视模型而定 | **tierflow-infer**（本仓库） |

调用链：

```
浏览器 → tierflow-zh → tierflow-core (relay 路由层)
                              │  POST /internal/v1/classify
                              ▼
                       tierflow-infer (GPU 难度打分)
```

tierflow-core 拿到难度分后，按配置把请求分级路由到不同档位的上游模型。本服务**只打分，不转发上游**。

---

## 工作原理

1. tierflow-core 传入整段 chat `messages`。
2. `build_routing_input` 拼成统一输入，Qwen backbone **单次 forward**，通过 forward hook 抓取指定 `(layer, head)` 的注意力特征。
3. 5 个 CG-TabM 回归头（`swe` / `tool` / `gaia` / `task` / `prog`）各自打分，归一化到 `0-2`。
4. proto 原型语义相似度对 5 维分数加权，得到最终 `0-2` 分，再线性映射到 `0-10`。

五维路由（顺序固定，须与训练一致）：

| 路由键 | 中文 | proto label |
|--------|------|-------------|
| swe | 纠错 | swe |
| tool | 工具调用 | tool |
| gaia | 通用任务 | gaia |
| task | 任务拆解 | task |
| prog | 编程 | prog |

---

## 接口契约

### POST `/internal/v1/classify`

鉴权：请求头 `X-Inference-Secret`，值须等于 `INFERENCE_SERVICE_SECRET`。

请求体：

```json
{
  "messages": [{"role": "user", "content": "帮我重构这段函数"}],
  "profile_id": "optional-string",
  "request_id": "optional-id"
}
```

`messages` 必填（1–512 条，单条 content ≤ 100KB）。

响应体：

```json
{
  "request_id": "chat-ab12cd34ef56",
  "scores_0_2": {"纠错": 0.83, "工具调用": 0.41, "通用任务": 1.12, "任务拆解": 0.67, "编程": 1.55},
  "proto_weighted_0_2": 1.08,
  "total_score_0_10": 5.4,
  "score_source": "proto_weighted_0_2",
  "latency_ms": 142.3
}
```

### GET `/ready`

无鉴权就绪探针，返回 `{"status":"ok","service":"tierflow-infer"}`。compose healthcheck 即探测此端点。

### GET `/internal/logs`

HMAC 签名鉴权（仅 tierflow-core 可调用），读取内存环形缓冲区日志。

---

## 前置要求

- **NVIDIA GPU** + 驱动（Qwen2.5-7B 以 bf16 加载，约需 16GB+ 显存）
- 本地**模型权重**（见下文「模型权重」一节，不随仓库分发）
- Docker 部署：Docker 24+、Docker Compose v2、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- 手动部署：Python 3.10+（镜像用 3.12）、[uv](https://docs.astral.sh/uv/)、与 GPU 匹配的 CUDA 版 PyTorch

---

## 模型权重

权重文件路径集中在 `config/model_paths.json` 管理，默认指向：

```
/root/autodl-tmp/model/
├── backbone/Qwen2.5-7B-Instruct/      # Qwen 骨干
└── 5roads/                            # 5 个 CG-TabM 路由头 + proto
    ├── swe/   tool/   gaia/   task/   prog/
    └── proto_artifact_v4.npz
```

部署到新机器时，二选一：

1. 把权重放到与 `config/model_paths.json` 一致的路径；或
2. 改 `config/model_paths.json` 内的路径指向实际位置。

> ML 安全：scaler 走 `_RestrictedScalerUnpickler` 白名单反序列化，PyTorch 用
> `weights_only=True`，proto 用 `allow_pickle=False`。请勿替换为未经审计的权重文件。

---

## 环境变量

复制模板并填值：

```bash
cp .env.example .env
```

**必填**（缺失或为占位符则启动失败）：

| 变量 | 说明 | 生成方式 |
|------|------|----------|
| `INTERNAL_SECRET` | 内部服务签名密钥，≥32 字符，须与 tierflow-core 一致 | `openssl rand -hex 32` |
| `INFERENCE_SERVICE_SECRET` | classify 端点鉴权密钥，须与 tierflow-core 的同名变量一致 | `openssl rand -hex 16` |

常用：

| 变量 | 说明 | 默认 |
|------|------|------|
| `ENV` / `DEBUG` | 运行环境 / 调试（`/docs` 仅 DEBUG 开放） | `production` / `false` |
| `INFERENCE_HOST` | 监听地址 | `0.0.0.0` |
| `INFERENCE_ALLOW_INSECURE_DEV` | 跳过 classify 鉴权（仅 `ENV != production` 生效） | `false` |
| `ROUTER_MODEL_PATHS` | `model_paths.json` 路径 | `config/model_paths.json` |
| `CUDA_VISIBLE_DEVICES` | 使用的 GPU 编号 | `0` |
| `GPU_CONCURRENCY_LIMIT` | GPU 推理并发上限（Semaphore） | `8` |
| `LOG_*` | 日志级别 / 文件 / 环形缓冲 | 见 `.env.example` |

> 🔒 `INFERENCE_SERVICE_SECRET` 与 `INTERNAL_SECRET` 必须和 tierflow-core 后端节点的取值完全一致，否则鉴权失败。`.env` 含密钥，已在 `.gitignore` 中，切勿提交。

---

## Docker 部署（推荐）

镜像基于 `python:3.12-slim`，需通过 NVIDIA Container Toolkit 透传 GPU（compose 已声明 `deploy.resources` GPU 预留）。模型权重与 `config/` 以**只读卷**挂载，不打进镜像。

```bash
# 1. 配置环境变量
cp .env.example .env && vim .env        # 至少填 INTERNAL_SECRET + INFERENCE_SERVICE_SECRET

# 2. 指定宿主机模型权重目录（默认 /srv/eucal/models）
export MODEL_WEIGHTS_HOST_PATH=/root/autodl-tmp/model

# 3. 构建并启动
docker compose build
docker compose up -d
```

compose 关键挂载（见 `docker-compose.yml`）：

| 宿主机 | 容器内 | 模式 | 说明 |
|--------|--------|------|------|
| `${MODEL_WEIGHTS_HOST_PATH:-/srv/eucal/models}` | `/app/models` | ro | 模型权重 |
| `./config` | `/app/config` | ro | `model_paths.json` |
| `tierflow_infer_logs`（named volume） | `/app/logs` | rw | 日志 |

> ⚠️ 容器内权重挂载在 `/app/models`，而仓库自带的 `config/model_paths.json` 默认指向
> `/root/autodl-tmp/model/...`。容器化部署时，请确保挂载路径与 `model_paths.json` 中的路径
> 对得上 —— 要么把权重挂到 `/root/autodl-tmp/model`，要么改 `model_paths.json` 指向 `/app/models`。

启动后 GPU 加载较慢，healthcheck `start_period` 设为 120s 等待模型就绪。

---

## 手动部署（无 Docker）

```bash
# 1. 安装依赖
uv sync                       # 生产可用 uv sync --no-dev

# 2. 配置环境变量
cp .env.example .env && vim .env

# 3. 校验环境（检查必填密钥）
uv run check-env

# 4. 确认 config/model_paths.json 中的权重路径在本机存在

# 5. 启动（GPU 推理服务固定单 worker）
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1
```

> 单 worker：模型权重占显存大，多 worker 会重复加载并引发 GPU 竞争。横向扩容请起多个独立实例（各占一张卡），由 tierflow-core 侧做负载分发。

生产建议用 systemd / supervisor 托管进程并设 `Restart=always`。

---

## 部署后验证

```bash
# 就绪探针（模型加载完成后才返回 200）
curl http://localhost:8001/ready

# classify 冒烟测试（替换为真实 INFERENCE_SERVICE_SECRET）
curl -X POST http://localhost:8001/internal/v1/classify \
  -H "Content-Type: application/json" \
  -H "X-Inference-Secret: <INFERENCE_SERVICE_SECRET>" \
  -d '{"messages":[{"role":"user","content":"写个快速排序"}]}'
```

接着在 tierflow-core 后端节点确认 `INFERENCE_SERVICE_URL` 指向本节点（默认 `http://127.0.0.1:8001`，跨节点部署时改成 `http://<GPU_IP>:8001`），且两端 `INFERENCE_SERVICE_SECRET` 一致。

---

## 故障排查

| 现象 | 排查方向 |
|------|----------|
| 启动报 `INTERNAL_SECRET must be configured` / `length must be at least 32` | `.env` 未填或长度不足，`openssl rand -hex 32` 重新生成 |
| classify 返回 403 forbidden | `X-Inference-Secret` 与 `INFERENCE_SERVICE_SECRET` 不一致 |
| classify 返回 503 推理服务未配置 | 生产环境未设 `INFERENCE_SERVICE_SECRET`（`INFERENCE_ALLOW_INSECURE_DEV` 在 production 下不生效） |
| 启动报找不到模型 / `FileNotFoundError` | `config/model_paths.json` 路径与实际权重位置不符 |
| 容器起不来 / 无 GPU | 确认已装 NVIDIA Container Toolkit，`docker run --rm --gpus all nvidia/cuda nvidia-smi` 验证 |
| `/ready` 长时间 503 | 模型仍在加载（7B + 5 头较慢），看 `docker compose logs -f` 等 `service_ready` |
| `blocked unpickle of ...` | scaler 文件包含白名单外的类，确认权重来源可信后再按提示扩白名单 |

---

## 开发命令

```bash
ruff check app/                              # lint
python -c "from app.main import create_app"  # 语法/导入检查
uv run check-env                             # 环境校验
```

