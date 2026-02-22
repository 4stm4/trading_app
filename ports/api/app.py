"""
FastAPI application factory and route wiring.
"""

from fastapi import FastAPI

from .controllers import router


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Trading System API",
        version="1.0.0",
    )
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ports.api.app:app", host="0.0.0.0", port=5000, reload=False)
