# Inference-Service 开发规范

## 项目概述

FastAPI GPU 推理微服务，运行 Qwen2.5-7B backbone + 5 个 CG-TabM 路由头，对聊天消息进行难度分类并返回路由决策（tier 1-5 → 模型名）。

技术栈：Python 3.10+, FastAPI, PyTorch, Transformers, NumPy, scikit-learn, httpx, Pydantic v2

## 架构分层

```
Controllers (FastAPI routers, 薄层)
    ↓ Depends(require_inference_secret) 鉴权
Services (stateless @staticmethod, 业务编排)
    ↓
Router Engine (HybridIntegratedDifficultyRouter, ML 推理核心)
    ↓
NN Models (CG-TabM regressors, PyTorch)
```

跨服务通信：
```
Services → Gateways (BaseGateway 子类)
    ↓
common/internal.py (HMAC 签名 + 熔断 + 重试 + 连接池)
```

配置来源：
```
admin-service (primary) → cached_previous → local_fallback (runtime_config.json)
```

## 核心规范

### Gateway 层

- 所有 Gateway 必须继承 `common.gateway.base.BaseGateway`
- 在 `__init__` 中声明 `base_url`、`timeout`、`error_map`
- 业务方法直接调用 `self._get()` / `self._post()` / `self._request()`，不要手写 try/except
- Gateway 实例在 lifespan 中创建，通过构造函数注入到 ConfigManager

```python
class AdminConfigGateway(BaseGateway):
    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            "admin-service",
            base_url=settings.ADMIN_SERVICE_URL,
            timeout=settings.CONFIG_FETCH_TIMEOUT_SECONDS,
        )

    async def fetch_active_config(self) -> dict | None:
        return await self._get("/api/v1/internal/routing-config/active/inference", allow_404=True)
```

### HTTP Client

- 禁止在业务代码中 `async with httpx.AsyncClient(...) as client:` 创建临时 client
- 使用 `common.internal.get_internal_client(base_url, timeout=...)` 获取共享连接池 client
- 连接池在 `main.py` lifespan shutdown 中通过 `close_internal_clients()` 关闭

### Service 层

- 使用 `@staticmethod`，依赖通过参数传入
- GPU 并发控制使用模块级 `asyncio.Semaphore`（通过 `init_gpu_semaphore()` 初始化）
- GPU 推理必须使用 `await asyncio.to_thread(engine.predict_chat_messages, ...)` 避免阻塞 event loop

### Controller 层

- 保持薄层，只做参数提取、调用 service、构造 response
- 鉴权通过 `Depends(require_inference_secret)`
- 依赖注入通过 `Depends(get_engine)` / `Depends(get_config_manager)`

### 异步规范

- 所有 handler 都是 `async def`
- 禁止在 async 上下文中调用阻塞函数（PyTorch forward、文件 IO、同步 HTTP）
- 阻塞操作使用 `await asyncio.to_thread(blocking_fn, ...)`
- 禁止在 per-request 路径中调用 `torch.cuda.empty_cache()`（会导致并发竞态）
- PyTorch 的 caching allocator 配合 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已能高效管理显存

### DI 模式

- 单例资源（engine、config_manager）使用模块级 globals + `set_xxx()` / `get_xxx()` 函数
- 在 lifespan startup 中初始化，通过 `Depends()` 注入到 controller
- GPU semaphore 使用模块级变量 + `init_gpu_semaphore(limit)` 初始化

### 日志规范

- 模块级 logger：`logger = logging.getLogger("inference_service")`
- 结构化日志使用 `log_event(logger, level, "eventName", key=value)`
- 禁止内联 `logging.getLogger("xxx").info(...)`
- 敏感信息（API key、密码）不得出现在日志中，observability 层已有自动脱敏
- 每次 classify 请求记录：latency、queue time、routing tier、config version/source、soft timeout

### 配置规范

- 所有配置通过 `inference_service.core.config.get_settings()` 单例访问
- 新增配置项加到 `InferenceSettings` 类并在 `.env.example` 中文档化
- 运行时路由配置通过 `ConfigManager` 管理（3-tier 降级：admin → cached → local）
- 生产环境敏感配置（密钥）必须通过环境变量注入
- `INFERENCE_ALLOW_INSECURE_DEV` 仅在 `ENV != "production"` 时生效

### 错误处理

- 业务异常使用 `inference_service.core.exceptions` 中的层级异常类：
  - `InferenceAuthError` (403) — 鉴权失败
  - `InferenceConfigError` (503) — 配置不可用
  - `InferenceUnavailableError` (503) — 引擎未初始化
  - `InferenceTimeoutError` (504) — 推理超时
- Gateway 错误通过 `BaseGateway._handle_error()` + `error_map` 自动映射
- 禁止裸 `except Exception` 吞掉错误（除非有明确的降级逻辑并记录日志）

### ML 安全

- Pickle 加载必须使用 `_RestrictedScalerUnpickler` 白名单机制
- PyTorch 模型加载使用 `torch.load(..., weights_only=True)`
- NumPy 加载使用 `np.load(..., allow_pickle=False)`
- 模型文件路径通过 `config/model_paths.json` 集中管理

### 资源清理

- lifespan shutdown 中必须按顺序清理：
  1. `config_mgr.stop()` — 停止配置轮询
  2. `engine.cleanup()` — 清理临时目录
  3. `close_internal_clients()` — 关闭 HTTP 连接池

## 命令

```bash
# 开发启动
cd services/inference-service
PYTHONPATH=src uvicorn inference_service.main:app --host 0.0.0.0 --port 8004 --reload

# Lint
ruff check src/

# 语法检查
python -c "from inference_service.main import create_app"
```

## 文件命名

- Controller: `controllers/{domain}.py`
- Service: `services/{domain}_service.py`
- Gateway: `gateways/{target_domain}.py`
- Schema: `schemas/{domain}.py`
- NN Model: `nn/{model_name}.py`
- Utils: `utils/{function_group}.py`
- Config: `core/config.py`
- Dependencies: `core/dependencies.py`
- Exceptions: `core/exceptions.py`

## 端点

| 方法 | 路径 | 用途 | 鉴权 |
|------|------|------|------|
| POST | `/internal/v1/classify` | 难度分类 + 路由决策 | `X-Inference-Secret` |
| GET | `/ready` | 健康检查 | 无 |
| GET | `/internal/logs` | 日志环形缓冲区读取 | HMAC 签名 (admin-service only) |
