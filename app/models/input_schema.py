"""
Input Schema - Contract from Agent 2 (Architect)
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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
    responsibilities: list[str] = Field(..., min_items=1)
    dependencies: Optional[list[str]] = Field(default_factory=list)
    communication_patterns: list[CommunicationPattern] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate subdomain name format"""
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Name must be alphanumeric (- and _ allowed)")
        return v.lower()


class ArchitectureInput(BaseModel):
    """Complete input from Agent 2"""

    project_name: str = Field(..., min_length=3)
    project_description: str
    subdomains: list[Subdomain] = Field(..., min_items=1)
    global_constraints: Optional[dict[str, str]] = Field(default_factory=dict)
    technical_stack: Optional[dict[str, str]] = Field(default_factory=dict)

class Config:
        json_schema_extra = {
            "example": {
                "project_name": "aircut-media-platform",
                "project_description": "Visual discovery engine for Aircut allowing barbers to showcase work.",
                "subdomains": [
                    {
                        "name": "portfolio-service",
                        "type": "core",
                        "description": "Central hub for barber galleries, style tagging, and visual content management.",
                        "bounded_context": "PortfolioContext",
                        "responsibilities": [
                            "Upload and optimize haircut images",
                            "Manage style categorization (Fade, Beard, Scissor Cut)",
                            "Handle 'Before & After' comparison sets",
                            "Index styles for visual discovery feed"
                        ],
                        "dependencies": ["identity-service", "analytics-service"],
                        "communication_patterns": ["sync_rest", "async_event"]
                    }
                ],
                "global_constraints": {
                    "max_response_time": "200ms",
                    "content_moderation": "automated_check",
                    "availability": "99.9%"
                },
                "technical_stack": {
                    "language": "Python",
                    "framework": "FastAPI",
                    "database": "MongoDB",  
                    "storage": "AWS S3"
                }
            }
        }