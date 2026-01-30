"""
Output Schema - Structured Functional Specifications

This schema remains unchanged as it defines the output contract.
The context chaining ensures consistency between Requirements, Events, APIs, and NFRs.

Context Chaining Flow:
1. Functional Requirements (base layer)
2. Domain Events (derived from requirements)
3. API Endpoints (derived from requirements + events)
4. Non-Functional Requirements (derived from full complexity analysis)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl


class RequirementType(str, Enum):
    """Types of requirements"""

    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    TECHNICAL = "technical"


class RequirementPriority(str, Enum):
    """Priority levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Requirement(BaseModel):
    """Single functional requirement"""

    id: str = Field(..., description="Unique requirement ID")
    type: RequirementType
    priority: RequirementPriority
    title: str = Field(..., min_length=5)
    description: str = Field(..., min_length=20)
    acceptance_criteria: list[str] = Field(..., min_items=1)
    related_requirements: Optional[list[str]] = Field(default_factory=list)


class EventDefinition(BaseModel):
    """
    Event published/subscribed by microservice.
    
    Context: Generated based on Functional Requirements to ensure alignment.
    Example: If requirement is "Create Order", event is "OrderCreatedEvent"
    """

    event_name: str = Field(..., pattern="^[A-Z][a-zA-Z0-9]*Event$")
    event_type: str = Field(..., description="Domain event type")
    payload_schema: dict[str, Any] = Field(..., description="JSON Schema")
    trigger_conditions: list[str]
    consumers: Optional[list[str]] = Field(default_factory=list)


class APIEndpoint(BaseModel):
    """
    REST API endpoint specification.
    
    Context: Generated based on Functional Requirements and Domain Events.
    Endpoints implement requirements and may trigger events.
    """

    method: str = Field(..., pattern="^(GET|POST|PUT|PATCH|DELETE)$")
    path: str = Field(..., pattern="^/.*")
    description: str
    request_schema: Optional[dict[str, Any]] = None
    response_schema: dict[str, Any]
    authentication_required: bool = True
    rate_limit: Optional[str] = "100/min"


class MessageQueueConfig(BaseModel):
    """Message queue configuration"""

    queue_name: str
    exchange_name: Optional[str] = None
    routing_key: Optional[str] = None
    message_schema: dict[str, Any]
    dlq_enabled: bool = True
    retry_policy: Optional[dict[str, int]] = None


class ServiceDependency(BaseModel):
    """Dependency on another service"""

    service_name: str
    dependency_type: str = Field(..., description="database|service|external_api")
    communication_method: str = Field(..., description="rest|grpc|event|queue")
    criticality: str = Field(..., pattern="^(critical|high|medium|low)$")
    fallback_strategy: Optional[str] = None


class MicroserviceSpec(BaseModel):
    """
    Complete functional specification for a microservice.
    
    All components are generated with context chaining:
    - Events align with Requirements
    - APIs align with Requirements and Events
    - NFRs reflect the complexity of all components
    """

    service_name: str = Field(..., min_length=3)
    version: str = "1.0.0"
    description: str = Field(..., min_length=20)
    bounded_context: str

    # Requirements
    functional_requirements: list[Requirement]
    non_functional_requirements: list[Requirement]

    # Communication
    events_published: list[EventDefinition] = Field(default_factory=list)
    events_subscribed: list[EventDefinition] = Field(default_factory=list)
    api_endpoints: list[APIEndpoint] = Field(default_factory=list)
    message_queues: list[MessageQueueConfig] = Field(default_factory=list)

    # Dependencies
    dependencies: list[ServiceDependency] = Field(default_factory=list)

    # Technical
    technology_stack: dict[str, str]
    infrastructure_requirements: dict[str, Any]
    monitoring_requirements: list[str]

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: str = "agent3-spec-generator"


class FunctionalSpecificationOutput(BaseModel):
    """Complete output for Agent 4"""

    project_name: str
    project_description: str
    specification_version: str = "1.0.0"

    microservices: list[MicroserviceSpec]

    # Global Architecture
    inter_service_communication: dict[str, list[str]] = Field(
        default_factory=dict, description="Service-to-service communication map"
    )
    shared_infrastructure: dict[str, Any] = Field(
        default_factory=dict, description="Databases, caches, message brokers"
    )

    # Documentation
    architecture_diagram_url: Optional[HttpUrl] = None
    api_documentation_url: Optional[HttpUrl] = None

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = "1.0.0"

    class Config:
        json_schema_extra = {
            "example": {
                "project_name": "ecommerce-platform",
                "project_description": "Microservices e-commerce",
                "microservices": [
                    {
                        "service_name": "order-service",
                        "version": "1.0.0",
                        "description": "Order management service",
                        "bounded_context": "Sales",
                        "functional_requirements": [
                            {
                                "id": "FR-001",
                                "type": "functional",
                                "priority": "critical",
                                "title": "Create Order",
                                "description": "User can create new order",
                                "acceptance_criteria": ["Order ID generated", "Payment initiated"],
                            }
                        ],
                        "non_functional_requirements": [],
                        "events_published": [
                            {
                                "event_name": "OrderCreatedEvent",
                                "event_type": "domain",
                                "payload_schema": {"order_id": "string"},
                                "trigger_conditions": ["Order validated"],
                                "consumers": ["inventory-service"],
                            }
                        ],
                        "api_endpoints": [],
                        "dependencies": [],
                        "technology_stack": {"language": "Python"},
                        "infrastructure_requirements": {},
                        "monitoring_requirements": ["Prometheus metrics"],
                    }
                ],
            }
        }