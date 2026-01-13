"""
FastAPI Server - REST API for Specification Generation
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.llm_config import llm_engine
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
    # Startup
    logger.info("Initializing Agent 3 - Functional Specification Generator")
    try:
        llm_engine.initialize()
        logger.info(f"LLM Engine initialized: {llm_engine.provider}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Agent 3")


# Initialize FastAPI
app = FastAPI(
    title="Agent 3 - Functional Specification Generator",
    description="Generates detailed functional specifications from architectural designs",
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
        "llm_provider": llm_engine.provider,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Verify LLM is accessible
        engine = llm_engine.get_engine()
        return {
            "status": "healthy",
            "llm_status": "connected" if engine else "disconnected",
            "service": settings.service_name,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service unhealthy"
        )


@app.post(
    "/generate",
    response_model=FunctionalSpecificationOutput,
    status_code=status.HTTP_200_OK,
    tags=["Generation"],
)
async def generate_specifications(architecture: ArchitectureInput):
    """
    Generate functional specifications from architecture.

    Args:
        architecture: Validated architecture input from Agent 2

    Returns:
        Complete functional specification for all microservices
    """
    try:
        logger.info(f"Generating specs for project: {architecture.project_name}")
        logger.info(f"Processing {len(architecture.subdomains)} subdomains")

        # Generate specifications
        result = orchestrator.generate_all_specs(architecture)

        # Validate output schema
        output = FunctionalSpecificationOutput(**result)

        logger.info(f"Successfully generated {len(output.microservices)} microservice specs")
        return output

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


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
        timeout_keep_alive=300,
    )
