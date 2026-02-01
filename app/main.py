"""
FastAPI Server - TRUE SSE Streaming Version
FIXED: Optimized for real-time streaming without buffering
"""

import logging
import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.models.input_schema import ArchitectureInput
from app.models.output_schema import FunctionalSpecificationOutput
from app.modules.generator_module import SpecificationOrchestrator

# Configure logging with more detailed format
logging.basicConfig(
    level=getattr(logging, settings.api_log_level.upper()),
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events with enhanced diagnostics"""
    logger.info("=" * 70)
    logger.info("🚀 Initializing Agent 3 - Functional Specification Generator")
    logger.info("=" * 70)
    logger.info(f"📍 Ollama URL: {settings.ollama_base_url}")
    logger.info(f"🤖 Model: {settings.ollama_model}")
    logger.info(f"🌊 Streaming Mode: TRUE SSE (asyncio.to_thread)")
    logger.info(f"🔧 Environment: {settings.environment if hasattr(settings, 'environment') else 'production'}")
    
    # Test Ollama connection with detailed diagnostics
    try:
        import httpx
        logger.info("🔍 Testing Ollama connection...")
        
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{settings.ollama_base_url}/api/tags")
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"✅ Ollama connection successful")
                logger.info(f"📋 Available models: {len(models)}")
                
                # Check if configured model exists
                model_names = [m.get('name', '') for m in models]
                if settings.ollama_model in model_names:
                    logger.info(f"✅ Model '{settings.ollama_model}' is available")
                else:
                    logger.warning(f"⚠️ Model '{settings.ollama_model}' not found in Ollama")
                    logger.warning(f"Available models: {', '.join(model_names[:5])}")
            else:
                logger.warning(f"⚠️ Ollama responded with status {response.status_code}")
                
    except httpx.ConnectError as e:
        logger.error(f"❌ Ollama connection failed: Cannot connect to {settings.ollama_base_url}")
        logger.error(f"   Make sure Ollama is running: ollama serve")
        logger.warning("⚠️ Service will start but generation will fail")
    except Exception as e:
        logger.error(f"❌ Ollama connection error: {e}")
        logger.warning("⚠️ Service will start but generation may fail")

    logger.info("=" * 70)
    logger.info("✅ Agent 3 ready to accept requests")
    logger.info("=" * 70)
    
    yield
    
    logger.info("=" * 70)
    logger.info("👋 Shutting down Agent 3")
    logger.info("=" * 70)


# Initialize FastAPI with enhanced configuration
app = FastAPI(
    title="Agent 3 - Functional Specification Generator",
    description="""
    Generates detailed functional specifications from architectural designs.
    
    **Features:**
    - Real-time SSE streaming for progress updates
    - DSPy-powered LLM generation with Chain-of-Thought
    - Non-blocking async processing with thread pool
    - Comprehensive error handling and fallbacks
    
    **Streaming Events:**
    - `start`: Generation initiated
    - `progress`: Real-time updates on each step
    - `microservice`: Complete microservice specification
    - `complete`: Final output with all specifications
    - `error`: Error information if something fails
    """,
    version=settings.service_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration with explicit settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Important for SSE
)

# Initialize orchestrator (singleton pattern)
orchestrator = SpecificationOrchestrator()


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "operational",
        "streaming": {
            "enabled": True,
            "type": "SSE (Server-Sent Events)",
            "implementation": "asyncio.to_thread + EventSourceResponse"
        },
        "architecture": "streaming-only",
        "llm": {
            "provider": "ollama",
            "url": settings.ollama_base_url,
            "model": settings.ollama_model
        },
        "endpoints": {
            "generate": "/generate/stream (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)",
            "schemas": {
                "input": "/schemas/input (GET)",
                "output": "/schemas/output (GET)"
            }
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint with detailed Ollama verification
    
    Returns comprehensive health status including:
    - Service status
    - Ollama connectivity
    - Model availability
    """
    health_status = {
        "status": "healthy",
        "service": settings.service_name,
        "version": settings.service_version,
        "timestamp": asyncio.get_event_loop().time(),
    }
    
    # Check Ollama connectivity
    try:
        import httpx
        
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{settings.ollama_base_url}/api/tags")
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                health_status["ollama"] = {
                    "status": "connected",
                    "url": settings.ollama_base_url,
                    "models_available": len(models),
                    "configured_model": settings.ollama_model,
                    "model_exists": settings.ollama_model in model_names
                }
            else:
                health_status["ollama"] = {
                    "status": "error",
                    "url": settings.ollama_base_url,
                    "http_status": response.status_code
                }
                health_status["status"] = "degraded"
                
    except httpx.ConnectError:
        health_status["ollama"] = {
            "status": "disconnected",
            "url": settings.ollama_base_url,
            "error": "Cannot connect to Ollama. Is it running?"
        }
        health_status["status"] = "degraded"
        
    except Exception as e:
        health_status["ollama"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    return health_status


@app.post("/generate/stream", tags=["Generation"])
async def generate_specifications_stream(
    architecture: ArchitectureInput,
    request: Request,
    skip_details: bool = False
):
    """
    Generate functional specifications with TRUE SSE streaming.
    
    This endpoint provides real-time progress updates as specifications are generated.
    
    **SSE Event Types:**
    - `start`: Initial event when generation begins
    - `progress`: Real-time updates on generation progress
      - Fields: subdomain, step, status, message, timestamp
    - `microservice`: Complete microservice specification
      - Contains: service_name, requirements, api_endpoints, events, etc.
    - `complete`: Final event with all specifications
      - Contains: project_name, microservices[], infrastructure, etc.
    - `error`: Error information if generation fails
    
    **Progress Steps:**
    1. `started` - Subdomain processing initiated
    2. `requirements` - Generating functional requirements
    3. `api` - Generating API endpoints
    4. `events` - Generating domain events
    5. `nfr` - Generating non-functional requirements
    6. `completed` - Subdomain processing finished
    
    Args:
        architecture: Validated architecture input from Agent 2
        request: FastAPI request object (for disconnect detection)
        skip_details: If True, skip detailed generation (for testing)
        
    Returns:
        EventSourceResponse with real-time SSE stream
        
    Example curl command:
        ```bash
        curl -N -H "Accept: text/event-stream" \
             -H "Content-Type: application/json" \
             -d @input.json \
             http://localhost:8000/generate/stream
        ```
    """
    
    async def event_generator():
        """
        SSE event generator with proper async handling
        
        KEY FEATURES:
        - Non-blocking event generation
        - Client disconnect detection
        - Comprehensive error handling
        - Immediate event flushing with asyncio.sleep(0.01)
        """
        try:
            # Log request details
            logger.info("=" * 70)
            logger.info(f"📦 New SSE stream request")
            logger.info(f"   Project: {architecture.project_name}")
            logger.info(f"   Subdomains: {len(architecture.subdomains)}")
            logger.info(f"   Skip Details: {skip_details}")
            logger.info(f"   Client: {request.client.host if request.client else 'unknown'}")
            logger.info("=" * 70)
            
            # Send initial event
            yield {
                "event": "start",
                "data": json.dumps({
                    "project_name": architecture.project_name,
                    "total_subdomains": len(architecture.subdomains),
                    "skip_details": skip_details,
                    "message": "🚀 Generation started",
                    "timestamp": asyncio.get_event_loop().time()
                })
            }
            
            # ✅ CRITICAL: Force immediate flush
            await asyncio.sleep(0.01)
            
            # Check if client is still connected
            if await request.is_disconnected():
                logger.warning("⚠️ Client disconnected before generation started")
                return
            
            # Generate specifications with streaming
            event_count = 0
            async for event in orchestrator.generate_all_specs_streaming(
                architecture,
                skip_details=skip_details
            ):
                # Check for client disconnect periodically
                if event_count % 5 == 0 and await request.is_disconnected():
                    logger.warning("⚠️ Client disconnected during generation")
                    return
                
                # Yield event
                yield event
                event_count += 1
                
                # ✅ CRITICAL: Force flush after each event
                await asyncio.sleep(0.01)
                
                # Log progress events (not microservice data to avoid spam)
                if event.get("event") == "progress":
                    data = json.loads(event.get("data", "{}"))
                    step = data.get("step", "unknown")
                    subdomain = data.get("subdomain", "unknown")
                    message = data.get("message", "")
                    logger.info(f"   📍 [{subdomain}] {step}: {message}")
            
            logger.info("=" * 70)
            logger.info(f"✅ SSE stream completed successfully")
            logger.info(f"   Total events sent: {event_count}")
            logger.info("=" * 70)
            
        except asyncio.CancelledError:
            logger.warning("⚠️ SSE stream cancelled (client disconnected)")
            raise
            
        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": "validation_error",
                    "message": str(e),
                    "timestamp": asyncio.get_event_loop().time()
                })
            }
            
        except Exception as e:
            logger.error(f"❌ Generation failed with exception: {type(e).__name__}")
            logger.error(f"   Message: {str(e)}")
            logger.error(f"   Full traceback:", exc_info=True)
            
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": "internal_error",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": asyncio.get_event_loop().time()
                })
            }
    
    # ✅ Use EventSourceResponse for proper SSE handling
    return EventSourceResponse(
        event_generator(),
        headers={
            # Additional headers for optimal streaming
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/generate/test", tags=["Testing"])
async def test_streaming_simple(request: Request):
    """
    Simple streaming test endpoint to verify SSE is working
    
    Returns 10 events with timestamps to test real-time delivery.
    If events arrive all at once, SSE is not configured correctly.
    """
    async def test_generator():
        """Generate test events"""
        for i in range(10):
            yield {
                "event": "test",
                "data": json.dumps({
                    "index": i,
                    "message": f"Test event {i}",
                    "timestamp": asyncio.get_event_loop().time()
                })
            }
            
            # ✅ Critical: Add delay to simulate work and force flush
            await asyncio.sleep(0.5)  # 500ms between events
            
            # Check disconnect
            if await request.is_disconnected():
                logger.info("Client disconnected from test stream")
                return
        
        # Final event
        yield {
            "event": "complete",
            "data": json.dumps({
                "message": "Test completed",
                "total_events": 10
            })
        }
    
    return EventSourceResponse(test_generator())


@app.get("/schemas/input", tags=["Documentation"])
async def get_input_schema():
    """
    Get JSON schema for input validation
    
    Returns the complete Pydantic schema for ArchitectureInput,
    which can be used to validate requests before sending them.
    """
    return ArchitectureInput.model_json_schema()


@app.get("/schemas/output", tags=["Documentation"])
async def get_output_schema():
    """
    Get JSON schema for output validation
    
    Returns the complete Pydantic schema for FunctionalSpecificationOutput,
    which describes the structure of the generated specifications.
    """
    return FunctionalSpecificationOutput.model_json_schema()


@app.get("/debug/settings", tags=["Debug"])
async def debug_settings():
    """
    Debug endpoint to check current settings
    
    WARNING: Disable in production or add authentication
    """
    return {
        "service_name": settings.service_name,
        "service_version": settings.service_version,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "api_reload": settings.api_reload,
        "api_log_level": settings.api_log_level,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting uvicorn server...")
    logger.info(f"   Host: {settings.api_host}")
    logger.info(f"   Port: {settings.api_port}")
    logger.info(f"   Reload: {settings.api_reload}")
    logger.info(f"   Workers: 1 (required for SSE streaming)")
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.api_log_level,
        timeout_keep_alive=600,  # 10 minutes for long-running streams
        workers=1,  # ✅ CRITICAL: Must be 1 for SSE to work properly
    )