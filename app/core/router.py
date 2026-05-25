"""Central router aggregating all controllers."""

from fastapi import APIRouter

from app.controller.classify import router as classify_router

api_router = APIRouter()
api_router.include_router(classify_router)
