"""
Output Schema - Structured Functional Specifications
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


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
    acceptance_criteria: List[str] = Field(..., min_items=1)
    related_requirements: Optional[List[str]] = Field(default_factory=list)


class EventDefinition(BaseModel):
    """Event published/subscribed by microservice"""
    event_name: str = Field(..., pattern="^[A-Z][a-zA-Z0-9]*Event$")
    event_type: str = Field(..., description="Domain event type")
    payload_schema: Dict[str, Any] = Field(..., description="JSON Schema")
    trigger_conditions: List[str]
    consumers: Optional[List[str]] = Field(default_factory=list)


class APIEndpoint(BaseModel):
    """REST API endpoint specification"""
    method: str = Field(..., pattern="^(GET|POST|PUT|PATCH|DELETE)$")
    path: str = Field(..., pattern="^/.*")
    description: str
    request_schema: Optional[Dict[str, Any]] = None
    response_schema: Dict[str, Any]
    authentication_required: bool = True
    rate_limit: Optional[str] = "100/min"


class MessageQueueConfig(BaseModel):
    """Message queue configuration"""
    queue_name: str
    exchange_name: Optional[str] = None
    routing_key: Optional[str] = None
    message_schema: Dict[str, Any]
    dlq_enabled: bool = True
    retry_policy: Optional[Dict[str, int]] = None


class ServiceDependency(BaseModel):
    """Dependency on another service"""
    service_name: str
    dependency_type: str = Field(..., description="database|service|external_api")
    communication_method: str = Field(..., description="rest|grpc|event|queue")
    criticality: str = Field(..., pattern="^(critical|high|medium|low)$")
    fallback_strategy: Optional[str] = None


class MicroserviceSpec(BaseModel):
    """Complete functional specification for a microservice"""
    service_name: str = Field(..., min_length=3)
    version: str = "1.0.0"
    description: str = Field(..., min_length=20)
    bounded_context: str
    
    # Requirements
    functional_requirements: List[Requirement]
    non_functional_requirements: List[Requirement]
    
    # Communication
    events_published: List[EventDefinition] = Field(default_factory=list)
    events_subscribed: List[EventDefinition] = Field(default_factory=list)
    api_endpoints: List[APIEndpoint] = Field(default_factory=list)
    message_queues: List[MessageQueueConfig] = Field(default_factory=list)
    
    # Dependencies
    dependencies: List[ServiceDependency] = Field(default_factory=list)
    
    # Technical
    technology_stack: Dict[str, str]
    infrastructure_requirements: Dict[str, Any]
    monitoring_requirements: List[str]
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: str = "agent3-spec-generator"


class FunctionalSpecificationOutput(BaseModel):
    """Complete output for Agent 4"""
    project_name: str
    project_description: str
    specification_version: str = "1.0.0"
    
    microservices: List[MicroserviceSpec]
    
    # Global Architecture
    inter_service_communication: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Service-to-service communication map"
    )
    shared_infrastructure: Dict[str, Any] = Field(
        default_factory=dict,
        description="Databases, caches, message brokers"
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
                                "acceptance_criteria": ["Order ID generated", "Payment initiated"]
                            }
                        ],
                        "non_functional_requirements": [],
                        "events_published": [
                            {
                                "event_name": "OrderCreatedEvent",
                                "event_type": "domain",
                                "payload_schema": {"order_id": "string"},
                                "trigger_conditions": ["Order validated"],
                                "consumers": ["inventory-service"]
                            }
                        ],
                        "api_endpoints": [],
                        "dependencies": [],
                        "technology_stack": {"language": "Python"},
                        "infrastructure_requirements": {},
                        "monitoring_requirements": ["Prometheus metrics"]
                    }
                ]
            }
        }