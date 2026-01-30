"""
FastAPI Server - SSE Streaming Version
Streaming-only architecture to prevent timeouts
"""

import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.models.input_schema import ArchitectureInput
from app.models.output_schema import FunctionalSpecificationOutput
from app.modules.generator_module import SpecificationOrchestrator

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.api_log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Initializing Agent 3 - Functional Specification Generator")
    logger.info(f"📍 Ollama URL: {settings.ollama_base_url}")
    logger.info(f"🤖 Model: {settings.ollama_model}")
    
    # Test Ollama connection
    try:
        import httpx
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{settings.ollama_base_url}/api/tags")
            if response.status_code == 200:
                logger.info("✅ Ollama connection successful")
            else:
                logger.warning(f"⚠️ Ollama responded with status {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        logger.warning("⚠️ Service will start but generation may fail")

    yield
    logger.info("👋 Shutting down Agent 3")


# Initialize FastAPI
app = FastAPI(
    title="Agent 3 - Functional Specification Generator",
    description="Generates detailed functional specifications with SSE streaming",
    version=settings.service_version,
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = SpecificationOrchestrator()


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "operational",
        "streaming_enabled": True,
        "architecture": "streaming-only",
        "ollama_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint with Ollama verification"""
    try:
        import httpx
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_status = "connected" if response.status_code == 200 else "error"
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        ollama_status = "disconnected"
    
    return {
        "status": "healthy",
        "ollama_status": ollama_status,
        "service": settings.service_name,
        "version": settings.service_version,
    }


@app.post("/generate/stream", tags=["Generation"])
async def generate_specifications_stream(architecture: ArchitectureInput):
    """
    Generate functional specifications with SSE streaming.
    
    Returns a text/event-stream with real-time progress updates:
    - Event: progress → {"subdomain": "name", "step": "...", "status": "..."}
    - Event: microservice → {...complete microservice spec...}
    - Event: complete → {...final output...}
    - Event: error → {"error": "message"}
    
    Args:
        architecture: Validated architecture input from Agent 2
        
    Returns:
        SSE stream with generation progress and results
    """
    
    async def event_generator():
        """SSE event generator"""
        try:
            logger.info(f"📦 Starting SSE stream for: {architecture.project_name}")
            logger.info(f"🔧 Processing {len(architecture.subdomains)} subdomains")
            
            # Send initial event
            yield {
                "event": "start",
                "data": json.dumps({
                    "project_name": architecture.project_name,
                    "total_subdomains": len(architecture.subdomains),
                    "message": "Generation started"
                })
            }
            
            # Generate specifications with streaming callbacks
            async for event in orchestrator.generate_all_specs_streaming(architecture):
                yield event
            
            logger.info("✅ SSE stream completed successfully")
            
        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": "validation_error",
                    "message": str(e)
                })
            }
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": "internal_error",
                    "message": str(e)
                })
            }
    
    return EventSourceResponse(event_generator())


@app.get("/schemas/input", tags=["Documentation"])
async def get_input_schema():
    """Get JSON schema for input validation"""
    return ArchitectureInput.model_json_schema()


@app.get("/schemas/output", tags=["Documentation"])
async def get_output_schema():
    """Get JSON schema for output validation"""
    return FunctionalSpecificationOutput.model_json_schema()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.api_log_level,
        timeout_keep_alive=600,
    )