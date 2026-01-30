"""
Pytest Configuration - Comprehensive Fixtures
"""
#### TEST ARTICOLATI DA FARE


import json
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.input_schema import (
    ArchitectureInput,
    CommunicationPattern,
    Subdomain,
    SubdomainType,
)
from app.models.output_schema import (
    APIEndpoint,
    EventDefinition,
    MicroserviceSpec,
    Requirement,
    RequirementPriority,
    RequirementType,
    ServiceDependency,
)

# ==================== INPUT FIXTURES ====================


@pytest.fixture
def sample_subdomain() -> Subdomain:
    """Minimal valid subdomain"""
    return Subdomain(
        name="order-management",
        type=SubdomainType.CORE,
        description="Manages the complete order lifecycle from creation to fulfillment",
        bounded_context="Sales",
        responsibilities=[
            "Order creation and validation",
            "Order status tracking",
            "Order cancellation",
            "Integration with payment service",
        ],
        dependencies=["payment-service", "inventory-service"],
        communication_patterns=[CommunicationPattern.ASYNC_EVENT, CommunicationPattern.SYNC_REST],
    )


@pytest.fixture
def payment_subdomain() -> Subdomain:
    """Payment service subdomain"""
    return Subdomain(
        name="payment-service",
        type=SubdomainType.CORE,
        description="Handles payment processing and transaction management",
        bounded_context="Finance",
        responsibilities=[
            "Process payments",
            "Handle refunds",
            "Manage payment methods",
            "Transaction logging",
        ],
        dependencies=["fraud-detection"],
        communication_patterns=[CommunicationPattern.SYNC_REST],
    )


@pytest.fixture
def inventory_subdomain() -> Subdomain:
    """Inventory management subdomain"""
    return Subdomain(
        name="inventory-service",
        type=SubdomainType.SUPPORTING,
        description="Manages product inventory and stock levels",
        bounded_context="Warehouse",
        responsibilities=[
            "Track stock levels",
            "Reserve items",
            "Update inventory",
            "Low stock alerts",
        ],
        dependencies=[],
        communication_patterns=[CommunicationPattern.ASYNC_EVENT],
    )


@pytest.fixture
def sample_architecture(
    sample_subdomain: Subdomain, payment_subdomain: Subdomain
) -> ArchitectureInput:
    """Complete architecture with multiple subdomains"""
    return ArchitectureInput(
        project_name="ecommerce-platform",
        project_description="Modern microservices-based e-commerce platform with event-driven architecture",
        subdomains=[sample_subdomain, payment_subdomain],
        global_constraints={
            "max_response_time": "200ms",
            "availability": "99.9%",
            "data_retention": "7 years",
            "compliance": "PCI-DSS, GDPR",
        },
        technical_stack={
            "language": "Python",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "cache": "Redis",
            "message_broker": "RabbitMQ",
            "monitoring": "Prometheus",
        },
    )


@pytest.fixture
def minimal_architecture() -> ArchitectureInput:
    """Minimal valid architecture"""
    return ArchitectureInput(
        project_name="minimal-service",
        project_description="Minimal test service",
        subdomains=[
            Subdomain(
                name="test-service",
                type=SubdomainType.GENERIC,
                description="Test service for validation",
                bounded_context="Test",
                responsibilities=["Test functionality"],
            )
        ],
    )


# ==================== OUTPUT FIXTURES ====================


@pytest.fixture
def sample_requirement() -> Requirement:
    """Sample functional requirement"""
    return Requirement(
        id="FR-001",
        type=RequirementType.FUNCTIONAL,
        priority=RequirementPriority.CRITICAL,
        title="Create Order",
        description="User must be able to create a new order with multiple items",
        acceptance_criteria=[
            "Order ID is generated and returned",
            "Order is persisted in database",
            "OrderCreatedEvent is published",
            "Payment initiation is triggered",
        ],
        related_requirements=["FR-002", "NFR-001"],
    )


@pytest.fixture
def sample_event() -> EventDefinition:
    """Sample domain event"""
    return EventDefinition(
        event_name="OrderCreatedEvent",
        event_type="domain",
        payload_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "format": "uuid"},
                "customer_id": {"type": "string"},
                "total_amount": {"type": "number"},
                "items": {"type": "array"},
                "created_at": {"type": "string", "format": "date-time"},
            },
            "required": ["order_id", "customer_id", "total_amount"],
        },
        trigger_conditions=[
            "Order is validated",
            "Payment method is verified",
            "Items are in stock",
        ],
        consumers=["inventory-service", "notification-service", "analytics-service"],
    )


@pytest.fixture
def sample_api_endpoint() -> APIEndpoint:
    """Sample REST API endpoint"""
    return APIEndpoint(
        method="POST",
        path="/api/v1/orders",
        description="Create a new order",
        request_schema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "items": {"type": "array"},
                "payment_method": {"type": "string"},
            },
            "required": ["customer_id", "items"],
        },
        response_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "status": {"type": "string"},
                "created_at": {"type": "string"},
            },
        },
        authentication_required=True,
        rate_limit="100/min",
    )


@pytest.fixture
def sample_dependency() -> ServiceDependency:
    """Sample service dependency"""
    return ServiceDependency(
        service_name="payment-service",
        dependency_type="service",
        communication_method="rest",
        criticality="critical",
        fallback_strategy="Queue for later processing with exponential backoff",
    )


@pytest.fixture
def sample_microservice_spec(
    sample_requirement: Requirement,
    sample_event: EventDefinition,
    sample_api_endpoint: APIEndpoint,
    sample_dependency: ServiceDependency,
) -> MicroserviceSpec:
    """Complete microservice specification"""
    return MicroserviceSpec(
        service_name="order-service",
        version="1.0.0",
        description="Handles order management and lifecycle",
        bounded_context="Sales",
        functional_requirements=[sample_requirement],
        non_functional_requirements=[
            Requirement(
                id="NFR-001",
                type=RequirementType.NON_FUNCTIONAL,
                priority=RequirementPriority.HIGH,
                title="Response Time",
                description="API must respond within 200ms for 95th percentile",
                acceptance_criteria=["p95 < 200ms", "p99 < 500ms"],
            )
        ],
        events_published=[sample_event],
        events_subscribed=[],
        api_endpoints=[sample_api_endpoint],
        dependencies=[sample_dependency],
        technology_stack={
            "language": "Python 3.11",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "cache": "Redis",
        },
        infrastructure_requirements={
            "cpu": "2 cores",
            "memory": "4GB",
            "storage": "20GB",
            "replicas": 3,
        },
        monitoring_requirements=[
            "Prometheus metrics",
            "Structured logging",
            "Distributed tracing",
            "Health checks",
        ],
    )


# ==================== MOCK FIXTURES ====================


@pytest.fixture
def mock_llm_response() -> dict[str, str]:
    """Mock LLM response with valid JSON"""
    return {
        "functional_requirements": json.dumps(
            [
                {
                    "id": "FR-001",
                    "title": "Create Order",
                    "description": "User can create a new order with items",
                    "acceptance_criteria": [
                        "Order ID generated",
                        "Order stored in database",
                        "Event published",
                    ],
                    "priority": "critical",
                },
                {
                    "id": "FR-002",
                    "title": "Cancel Order",
                    "description": "User can cancel pending orders",
                    "acceptance_criteria": [
                        "Order status updated",
                        "Refund initiated",
                        "CancelEvent published",
                    ],
                    "priority": "high",
                },
            ]
        ),
        "non_functional_requirements": json.dumps(
            [
                {
                    "id": "NFR-001",
                    "title": "Performance",
                    "description": "Order creation must complete in < 200ms",
                    "acceptance_criteria": ["95th percentile < 200ms"],
                    "priority": "high",
                },
                {
                    "id": "NFR-002",
                    "title": "Availability",
                    "description": "Service must be available 99.9% of the time",
                    "acceptance_criteria": ["Uptime > 99.9%", "Max downtime 8.76h/year"],
                    "priority": "critical",
                },
            ]
        ),
        "events_published": json.dumps(
            [
                {
                    "event_name": "OrderCreatedEvent",
                    "event_type": "domain",
                    "payload_schema": {
                        "order_id": "string",
                        "customer_id": "string",
                        "total_amount": "number",
                    },
                    "trigger_conditions": ["Order validated and saved"],
                    "consumers": ["inventory-service", "notification-service"],
                }
            ]
        ),
        "events_subscribed": json.dumps(
            [
                {
                    "event_name": "PaymentCompletedEvent",
                    "event_type": "domain",
                    "payload_schema": {"payment_id": "string", "order_id": "string"},
                    "trigger_conditions": [],
                    "consumers": [],
                }
            ]
        ),
        "api_endpoints": json.dumps(
            [
                {
                    "method": "POST",
                    "path": "/api/v1/orders",
                    "description": "Create new order",
                    "request_schema": {"customer_id": "string", "items": "array"},
                    "response_schema": {"order_id": "string", "status": "string"},
                    "authentication_required": True,
                },
                {
                    "method": "GET",
                    "path": "/api/v1/orders/{order_id}",
                    "description": "Get order details",
                    "response_schema": {"order": "object"},
                    "authentication_required": True,
                },
            ]
        ),
        "service_dependencies": json.dumps(
            [
                {
                    "service_name": "payment-service",
                    "dependency_type": "service",
                    "communication_method": "rest",
                    "criticality": "critical",
                    "fallback_strategy": "Queue for retry",
                }
            ]
        ),
        "infrastructure_requirements": json.dumps(
            {
                "database": "PostgreSQL with replication",
                "cache": "Redis cluster",
                "storage": "S3 for documents",
                "queue": "RabbitMQ with DLQ",
            }
        ),
    }


@pytest.fixture
def mock_validation_response() -> dict[str, str]:
    """Mock validation response"""
    return {
        "validation_result": "Score: 92/100. All critical requirements met.",
        "recommendations": "Consider adding more detailed acceptance criteria for NFRs",
    }


@pytest.fixture
def mock_dspy_pipeline(mock_llm_response: dict, mock_validation_response: dict) -> Generator:
    """Mock DSPy pipeline"""
    with patch("app.modules.generator_module.dspy.ChainOfThought") as mock_cot:
        mock_generate = MagicMock()
        mock_generate.return_value = MagicMock(**mock_llm_response)

        mock_validate = MagicMock()
        mock_validate.return_value = MagicMock(**mock_validation_response)

        mock_cot.side_effect = [mock_generate, mock_validate]

        yield mock_cot


# ==================== API CLIENT FIXTURES ====================


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Test client for API testing"""
    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Mock authentication headers"""
    return {"Authorization": "Bearer mock_token_12345", "Content-Type": "application/json"}


# ==================== UTILITY FIXTURES ====================


@pytest.fixture
def temp_env_file(tmp_path):
    """Create temporary .env file"""
    env_content = """
OLLAMA_MODEL=mixtral:8x7b
API_PORT=8003
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    return env_file


@pytest.fixture
def sample_json_files(tmp_path):
    """Create sample JSON files for testing"""
    input_file = tmp_path / "input.json"
    input_file.write_text(
        json.dumps(
            {"project_name": "test-project", "project_description": "Test", "subdomains": []}
        )
    )

    return {"input": input_file}
