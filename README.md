# inference-service

GPU ML 推理服务：加载 Qwen2.5-7B 骨干模型和 5 个 CG-TabM 分类头，对用户 prompt 进行五维难度评分。

- 端口：8004
- 无数据库
- 需要 NVIDIA GPU 和模型权重文件

## 本地开发

```bash
# 1. 安装依赖
uv sync

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入模型路径、密钥等
# 确保 CUDA_VISIBLE_DEVICES 和模型权重路径正确

# 3. 环境检查
uv run check-env

# 4. 启动服务
uv run uvicorn inference_service.main:app --host 0.0.0.0 --port 8004 --reload
```

## 模型权重

模型路径在 `config/model_paths.json` 中配置，默认指向：

- 骨干模型：`/root/autodl-tmp/model/backbone/Qwen2.5-7B-Instruct`
- 分类头：`/root/autodl-tmp/model/5roads/` 下的 5 个子目录（swe、tool、gaia、task、prog）

## 认证

- 生产环境：请求需携带 `X-Inference-Secret` 头，值与 `INFERENCE_SERVICE_SECRET` 匹配
- 开发环境：设置 `INFERENCE_ALLOW_INSECURE_DEV=1` 跳过认证

## 健康检查

- `GET /health` — 存活探针
- `GET /ready` — 就绪探针
