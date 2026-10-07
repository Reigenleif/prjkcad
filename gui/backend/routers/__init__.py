try:
    from gui.backend.routers.health import router as health_router
    from gui.backend.routers.samples import router as samples_router
    from gui.backend.routers.chat import router as chat_router
    from gui.backend.routers.cad import router as cad_router
except ImportError:
    from routers.health import router as health_router
    from routers.samples import router as samples_router
    from routers.chat import router as chat_router
    from routers.cad import router as cad_router

__all__ = ["health_router", "samples_router", "chat_router", "cad_router"]
