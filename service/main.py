"""
Aify OpenMemory — Main FastAPI Application

Combines aify-container's orchestration with OpenMemory's memory intelligence.
The orchestrator IS the memory API. Qdrant and Neo4j are managed as sub-containers.
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.config import get_config
from service.routers import health, api, containers as containers_router


def _setup_logging(config):
    """Configure logging based on config."""
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    if config.log_format == "json":
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)


logger = logging.getLogger(__name__)


def _init_database():
    """Initialize SQLite database, create tables and default user/app."""
    import datetime
    from uuid import uuid4
    import os

    from service.database import Base, SessionLocal, engine
    from service.database.models import User, App

    # Create all tables
    Base.metadata.create_all(bind=engine)

    user_id = os.getenv("USER", "default_user")
    default_app_name = "openmemory"

    db = SessionLocal()
    try:
        # Create default user
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = User(
                id=uuid4(),
                user_id=user_id,
                name="Default User",
                created_at=datetime.datetime.now(datetime.UTC),
            )
            db.add(user)
            db.commit()
            logger.info(f"Created default user: {user_id}")

        # Create default app
        existing_app = db.query(App).filter(
            App.name == default_app_name, App.owner_id == user.id
        ).first()
        if not existing_app:
            app_obj = App(
                id=uuid4(),
                name=default_app_name,
                owner_id=user.id,
                created_at=datetime.datetime.now(datetime.UTC),
                updated_at=datetime.datetime.now(datetime.UTC),
            )
            db.add(app_obj)
            db.commit()
            logger.info(f"Created default app: {default_app_name}")
    finally:
        db.close()

    # Create brain_audit table
    try:
        from service.brain.tools import _ensure_audit_table
        _ensure_audit_table()
    except Exception as e:
        logger.warning(f"Brain audit table init: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle."""
    config = get_config()
    _setup_logging(config)
    logger.info(f"Starting {config.name} v{config.version}")

    # --- STARTUP ---

    # 1. Container manager (Qdrant, Neo4j, optional LLM containers)
    container_manager = None
    json_path = Path(config.config_dir) / "service.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {e}")
            config_data = {}

        if config_data.get("containers", {}).get("definitions"):
            from service.containers.manager import ContainerManager, load_container_definitions
            try:
                definitions, defaults = load_container_definitions(config_data)
                container_manager = ContainerManager(definitions, defaults)
                app.state.container_manager = container_manager
                await container_manager.start_background_tasks()
                logger.info(f"Container manager: {len(definitions)} containers defined")
            except Exception as e:
                logger.error(f"Container manager init failed: {e}")

    # 2. Initialize LLM backend (with container manager for URL resolution)
    try:
        from service.memory.llm import init_llm_backend
        init_llm_backend(container_manager)
        logger.info("LLM backend initialized")
    except Exception as e:
        logger.warning(f"LLM backend init failed: {e}")

    # 3. Initialize database (tables, default user/app)
    try:
        _init_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    # 4. Warm up memory client (connects to Qdrant + Neo4j)
    try:
        from service.memory.client import get_memory_client
        client = get_memory_client()
        if client:
            logger.info("Memory client initialized")
        else:
            logger.warning("Memory client unavailable (will retry on first use)")
    except Exception as e:
        logger.warning(f"Memory client warmup failed: {e}")

    # 5. Mount MCP server if enabled
    if config.mcp_enabled:
        from mcp_local.sse_server import setup_mcp_server
        setup_mcp_server(app)
        logger.info(f"MCP SSE at {config.mcp_path_prefix}/sse")

    yield

    # --- SHUTDOWN ---
    if container_manager:
        await container_manager.shutdown()
    logger.info(f"Shutting down {config.name}")


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title=config.name,
        version=config.version,
        description=config.description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    origins = config.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=("*" not in origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import brain router
    from service.routers import brain

    app.include_router(health.router)
    app.include_router(api.router, prefix="/api/v1")
    app.include_router(brain.router)
    app.include_router(containers_router.router)

    return app


app = create_app()
