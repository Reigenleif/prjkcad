import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [PROJECT_ROOT, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from gui.backend.cad_engine import DEFAULT_STL_DIR
    from gui.backend.routers import (
        health_router,
        samples_router,
        chat_router,
        cad_router,
    )
except ImportError:
    from cad_engine import DEFAULT_STL_DIR
    from routers import (
        health_router,
        samples_router,
        chat_router,
        cad_router,
    )

app = FastAPI(
    title="Text2CAD Studio API",
    description="Backend API for CAD Editor, DualSeq generation, and SolidWorks part manager",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DEFAULT_STL_DIR, exist_ok=True)
app.mount("/api/renders", StaticFiles(directory=DEFAULT_STL_DIR), name="renders")

app.include_router(health_router)
app.include_router(samples_router)
app.include_router(chat_router)
app.include_router(cad_router)

@app.get("/")
def root():
    return {"message": "Text2CAD Studio API is running", "docs": "/docs"}

@app.get("/health")
def root_health():
    return {"status": "online"}


FRONTEND_DIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(FRONTEND_DIST_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
