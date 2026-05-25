"""FastAPI app creation + CLI entry point for tierflow-infer."""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI

from app.common.components import C
from app.common.observability import configure_logging, get_logger, log_event
from app.core.config import get_settings, load_model_paths
from app.core.dependencies import set_config_manager, set_engine
from app.core.exceptions import install_inference_error_handlers
from app.core.router import api_router

_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROUTER_ASSETS = _SERVICE_ROOT / "config"

logger = get_logger(C.CORE_MAIN)


def _resolve_asset(setting_value: str, filename: str) -> str:
    if setting_value:
        return setting_value
    return str(_DEFAULT_ROUTER_ASSETS / filename)


def create_app() -> FastAPI:
    settings = get_settings()

    configure_logging(
        settings.LOG_LEVEL,
        service_name=settings.SERVICE_NAME,
        env=settings.ENV,
        log_dir=settings.LOG_DIR,
        enable_file_logging=settings.LOG_TO_FILE,
        file_prefix="tierflow_infer",
        file_max_bytes=settings.LOG_FILE_MAX_BYTES,
        file_backup_count=settings.LOG_FILE_BACKUP_COUNT,
        ring_buffer_capacity=settings.LOG_RING_BUFFER_CAPACITY,
    )

    model_paths_config = _resolve_asset(settings.ROUTER_MODEL_PATHS, "model_paths.json")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from app.common.internal import close_internal_clients
        from app.gateway.api_service_config import ApiServiceConfigGateway
        from app.service.classify_service import init_gpu_semaphore
        from app.service.config_manager import ConfigManager
        from app.service.router_engine import HybridIntegratedDifficultyRouter

        log_event(logger, logging.INFO, "service_starting", message="推理服务启动中")
        model_paths = load_model_paths(model_paths_config)

        gateway = ApiServiceConfigGateway()
        config_mgr = ConfigManager(
            gateway=gateway,
            refresh_interval_seconds=settings.CONFIG_REFRESH_INTERVAL_SECONDS,
        )
        await config_mgr.start()
        set_config_manager(config_mgr)

        engine = HybridIntegratedDifficultyRouter(
            model_paths,
            runtime_config=config_mgr.load(),
        )
        set_engine(engine)

        init_gpu_semaphore(settings.GPU_CONCURRENCY_LIMIT)

        log_event(logger, logging.INFO, "service_ready", message="推理服务就绪")
        yield
        await config_mgr.stop()
        engine.cleanup()
        await close_internal_clients()
        log_event(logger, logging.INFO, "service_stopped", message="推理服务已停止")

    from app.common.internal import build_internal_auth_dependency
    from app.common.internal_logs import build_internal_logs_router
    from app.common.observability import install_observability

    app = FastAPI(
        title="tierflow-infer",
        version=settings.VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    install_observability(app, service_name=settings.SERVICE_NAME)
    install_inference_error_handlers(app)

    app.include_router(api_router)

    _logs_auth = build_internal_auth_dependency(
        settings.INTERNAL_SECRET,
        allowed_callers={"tierflow-core"},
    )
    app.include_router(build_internal_logs_router(_logs_auth))

    @app.get("/ready")
    def ready() -> Dict[str, str]:
        return {"status": "ok", "service": settings.SERVICE_NAME}

    return app


app = create_app()


def cli() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="tierflow-infer")
    parser.add_argument("--host", type=str, default=settings.INFERENCE_HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    args = parser.parse_args()

    import uvicorn

    print("=" * 60)
    print("starting tierflow-infer")
    print(f"  host: {args.host}")
    print(f"  port: {args.port}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    cli()
