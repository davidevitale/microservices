"""
Input Schema - Contract from Agent 2 (Architect)
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
from enum import Enum


class SubdomainType(str, Enum):
    """Types of subdomains"""
    CORE = "core"
    SUPPORTING = "supporting"
    GENERIC = "generic"


class CommunicationPattern(str, Enum):
    """Communication patterns between services"""
    SYNC_REST = "sync_rest"
    ASYNC_EVENT = "async_event"
    MESSAGE_QUEUE = "message_queue"
    GRAPHQL = "graphql"


class Subdomain(BaseModel):
    """Single subdomain definition from architecture"""
    name: str = Field(..., min_length=3, max_length=100)
    type: SubdomainType
    description: str = Field(..., min_length=10)
    bounded_context: str = Field(..., description="DDD Bounded Context")
    responsibilities: List[str] = Field(..., min_items=1)
    dependencies: Optional[List[str]] = Field(default_factory=list)
    communication_patterns: List[CommunicationPattern] = Field(default_factory=list)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate subdomain name format"""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Name must be alphanumeric (- and _ allowed)")
        return v.lower()


class ArchitectureInput(BaseModel):
    """Complete input from Agent 2"""
    project_name: str = Field(..., min_length=3)
    project_description: str
    subdomains: List[Subdomain] = Field(..., min_items=1)
    global_constraints: Optional[Dict[str, str]] = Field(default_factory=dict)
    technical_stack: Optional[Dict[str, str]] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "project_name": "ecommerce-platform",
                "project_description": "Modern e-commerce system with microservices",
                "subdomains": [
                    {
                        "name": "order-management",
                        "type": "core",
                        "description": "Handles order lifecycle",
                        "bounded_context": "Sales",
                        "responsibilities": ["Order creation", "Order tracking"],
                        "dependencies": ["payment-service"],
                        "communication_patterns": ["async_event", "sync_rest"]
                    }
                ],
                "global_constraints": {
                    "max_response_time": "200ms",
                    "availability": "99.9%"
                },
                "technical_stack": {
                    "language": "Python",
                    "framework": "FastAPI"
                }
            }
        }