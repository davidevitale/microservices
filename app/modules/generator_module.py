"""
generator_module.py - DSPy-Based Specification Generator
WITH PYDANTIC VALIDATION - COMPLETE VERSION
Combines Pydantic models with robust parsing logic
"""

import json
import re
import asyncio
from datetime import datetime
from typing import Any, Optional, AsyncGenerator, Callable

import dspy
from pydantic import ValidationError

from app.models.input_schema import ArchitectureInput, Subdomain
from app.models.output_schema import (
    Requirement,
    RequirementType,
    RequirementPriority,
    APIEndpoint,
    EventDefinition,
    MicroserviceSpec,
    FunctionalSpecificationOutput
)
from app.core.llm_config import initialize_llm_engine


# ============================================================================
# GLOBAL CONSTANTS - Infrastructure and Monitoring Templates
# ============================================================================

SHARED_INFRASTRUCTURE = {
    "api_gateway": {
        "type": "Kong",
        "version": "3.x",
        "features": ["rate_limiting", "authentication", "logging"]
    },
    "message_broker": {
        "type": "RabbitMQ",
        "version": "3.12",
        "clustering": True
    },
    "service_mesh": {
        "type": "Istio",
        "version": "1.20",
        "features": ["traffic_management", "security", "observability"]
    },
    "monitoring": {
        "metrics": "Prometheus + Grafana",
        "logging": "ELK Stack",
        "tracing": "Jaeger"
    }
}

INFRASTRUCTURE_TEMPLATE = {
    "database": {
        "type": "PostgreSQL",
        "version": "15+",
        "replicas": 2,
        "backup_strategy": "daily",
        "connection_pool": 20
    },
    "cache": {
        "type": "Redis",
        "version": "7+",
        "ttl": "1h",
        "persistence": "AOF",
        "max_memory": "2GB"
    },
    "message_queue": {
        "type": "RabbitMQ",
        "version": "3.12+",
        "durable": True,
        "prefetch_count": 10,
        "max_priority": 10
    },
    "compute": {
        "cpu": "2 cores",
        "memory": "4GB",
        "storage": "20GB",
        "auto_scaling": True
    }
}

MONITORING_REQUIREMENTS = [
    "Prometheus metrics endpoint /metrics",
    "Health check endpoint /health",
    "Structured JSON logging",
    "Distributed tracing with Jaeger",
    "Error rate and latency monitoring"
]

DEFAULT_TECH_STACK = {
    "language": "Python",
    "framework": "FastAPI",
    "database": "PostgreSQL",
    "cache": "Redis",
    "message_broker": "RabbitMQ"
}


# ============================================================================
# SHARED UTILITY FUNCTIONS - DRY Compliance
# ============================================================================

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON from text that may contain markdown code blocks or other noise.
    
    FIXED: Properly handles LLM output that continues after JSON
    Example: [...JSON...] Inoltre, possiamo aggiungere...
    
    Args:
        text: Raw text that may contain JSON
        
    Returns:
        Extracted JSON string or None if no JSON found
    """
    # Try to match JSON inside markdown code blocks
    match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try to find JSON array or object
    # First, find the start of JSON
    json_start = -1
    for i, char in enumerate(text):
        if char in '[{':
            json_start = i
            break
    
    if json_start == -1:
        return text.strip()
    
    # Now find the matching closing bracket
    # We need to count nested brackets
    opening = text[json_start]
    closing = ']' if opening == '[' else '}'
    
    depth = 0
    for i in range(json_start, len(text)):
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                # Found the end of JSON
                return text[json_start:i+1]
    
    # If we couldn't find proper closing, try the old method as fallback
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        json_text = match.group(1)
        # Try to find last ] or } and cut there
        last_bracket = max(json_text.rfind(']'), json_text.rfind('}'))
        if last_bracket != -1:
            return json_text[:last_bracket+1]
        return json_text
    
    # Return cleaned text as fallback
    return text.strip()


# ============================================================================
# DSPy SIGNATURES - Input/Output Contracts
# ============================================================================

class GenerateFunctionalRequirements(dspy.Signature):
    """Generate structured functional requirements for a microservice"""
    
    service_name: str = dspy.InputField(desc="Nome del microservizio")
    service_description: str = dspy.InputField(desc="Descrizione dettagliata del servizio")
    responsibilities: str = dspy.InputField(desc="Responsabilità principali (lista)")
    bounded_context: str = dspy.InputField(desc="Bounded Context DDD")
    
    requirements_json: str = dspy.OutputField(
        desc="""Array JSON di requisiti funzionali. Formato ESATTO:
[
  {
    "id": "FR-001",
    "type": "functional",
    "priority": "critical|high|medium|low",
    "title": "Titolo conciso (max 100 char)",
    "description": "Descrizione dettagliata (min 50 char)",
    "acceptance_criteria": ["criterio 1", "criterio 2", "criterio 3"]
  }
]
Genera 3-5 requisiti. NO markdown, NO spiegazioni, SOLO JSON valido."""
    )


class GenerateAPIEndpoints(dspy.Signature):
    """Generate REST API endpoints specification"""
    
    service_name: str = dspy.InputField()
    functional_requirements: str = dspy.InputField(desc="Requisiti funzionali già generati")
    
    endpoints_json: str = dspy.OutputField(
        desc="""Array JSON di API endpoints. Formato:
[
  {
    "method": "GET|POST|PUT|DELETE|PATCH",
    "path": "/api/v1/resource",
    "description": "Descrizione endpoint",
    "request_schema": {"field": "type"},
    "response_schema": {"field": "type"},
    "authentication_required": true,
    "rate_limit": "100/min"
  }
]
Genera 3-5 endpoints REST. SOLO JSON valido."""
    )


class GenerateDomainEvents(dspy.Signature):
    """Generate domain events for event-driven architecture"""
    
    service_name: str = dspy.InputField()
    responsibilities: str = dspy.InputField()
    communication_patterns: str = dspy.InputField()
    
    events_json: str = dspy.OutputField(
        desc="""Array JSON di eventi di dominio. Formato:
[
  {
    "event_name": "EntityCreatedEvent",
    "event_type": "integration",
    "payload_schema": {"entity_id": "string", "timestamp": "string"},
    "trigger_conditions": ["condizione 1", "condizione 2"],
    "consumers": ["service-name"]
  }
]
Genera 2-4 eventi. SOLO JSON valido."""
    )


class GenerateNonFunctionalRequirements(dspy.Signature):
    """Generate non-functional requirements (NFRs)"""
    
    service_name: str = dspy.InputField()
    service_type: str = dspy.InputField(desc="core|supporting|generic")
    
    nfr_json: str = dspy.OutputField(
        desc="""Array JSON di requisiti non funzionali. Formato:
[
  {
    "id": "NFR-001",
    "type": "non_functional",
    "priority": "high|medium|low",
    "title": "Titolo (Performance|Security|Scalability|Availability)",
    "description": "Descrizione dettagliata",
    "acceptance_criteria": ["SLA specifico", "metrica misurabile"]
  }
]
Genera 3-4 NFRs. Focus: performance, sicurezza, scalabilità, disponibilità."""
    )


# ============================================================================
# GENERATOR MODULES - Encapsulated DSPy Modules
# ============================================================================

class FunctionalRequirementsGenerator(dspy.Module):
    """Module for generating functional requirements"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateFunctionalRequirements)
    
    def forward(self, subdomain: Subdomain, progress_callback: Optional[Callable] = None) -> list[Requirement]:
        """Generate functional requirements for a subdomain"""
        try:
            if progress_callback:
                progress_callback("Preparing functional requirements...")
            
            responsibilities_str = "\n- " + "\n- ".join(subdomain.responsibilities[:5])
            
            if progress_callback:
                progress_callback("Calling LLM for functional requirements...")
            
            result = self.generate(
                service_name=subdomain.name,
                service_description=subdomain.description,
                responsibilities=responsibilities_str,
                bounded_context=subdomain.bounded_context
            )
            
            if progress_callback:
                progress_callback("Parsing functional requirements...")
            
            requirements = self._parse_requirements(result.requirements_json)
            
            if requirements:
                if progress_callback:
                    progress_callback(f"✅ Generated {len(requirements)} functional requirements")
                return requirements
            else:
                if progress_callback:
                    progress_callback("⚠️ Using fallback requirements")
                return self._fallback_requirements(subdomain.name)
            
        except Exception as e:
            print(f"⚠️ Functional requirements generation failed: {e}")
            if progress_callback:
                progress_callback(f"⚠️ Using fallback requirements: {str(e)[:100]}")
            return self._fallback_requirements(subdomain.name)
    
    def _parse_requirements(self, json_str: str) -> list[Requirement]:
        """
        Parse functional requirements JSON with Pydantic validation
        """
        print(f"   📍 [FR Parser] Raw input length: {len(json_str)} chars")
        print(f"   📍 [FR Parser] First 200 chars: {json_str[:200]}")
        
        json_clean = extract_json_from_text(json_str)
        
        if not json_clean:
            print("   ❌ [FR Parser] No JSON found in text")
            return []
        
        print(f"   📍 [FR Parser] Cleaned JSON length: {len(json_clean)} chars")
        print(f"   📍 [FR Parser] Cleaned JSON preview: {json_clean[:300]}")
        
        try:
            data = json.loads(json_clean)
            print(f"   ✅ [FR Parser] JSON parsed successfully, type: {type(data)}")
            
            # Handle wrapped JSON
            if isinstance(data, dict):
                print(f"   📍 [FR Parser] Got dict with keys: {list(data.keys())}")
                for key in ['requirements', 'functional_requirements', 'items', 'data']:
                    if key in data and isinstance(data[key], list):
                        print(f"   ✅ [FR Parser] Found requirements in key '{key}'")
                        data = data[key]
                        break
                else:
                    print(f"   ⚠️ [FR Parser] Converting dict to list")
                    data = [data]
            
            if not isinstance(data, list):
                print(f"   ❌ [FR Parser] Data is not a list: {type(data)}")
                return []
            
            print(f"   📍 [FR Parser] Processing {len(data)} requirement items")
            
            validated = []
            for idx, req in enumerate(data[:5]):
                if not isinstance(req, dict):
                    print(f"   ⚠️ [FR Parser] Item {idx} is not a dict: {type(req)}")
                    continue
                
                print(f"   📍 [FR Parser] Item {idx} keys: {list(req.keys())}")
                
                try:
                    # Validate with Pydantic
                    requirement = Requirement(
                        id=req.get("id", f"FR-{len(validated)+1:03d}"),
                        type=RequirementType.FUNCTIONAL,
                        priority=RequirementPriority(req.get("priority", "medium").lower()),
                        title=str(req.get("title", ""))[:200],
                        description=str(req.get("description", ""))[:1000],
                        acceptance_criteria=req.get("acceptance_criteria", [])
                    )
                    
                    if not requirement.title:
                        print(f"   ⚠️ [FR Parser] Item {idx} missing title, skipping")
                        continue
                    
                    validated.append(requirement)
                    print(f"   ✅ [FR Parser] Item {idx} validated: {requirement.title}")
                    
                except (ValidationError, ValueError) as e:
                    print(f"   ⚠️ [FR Parser] Item {idx} validation failed: {e}")
                    continue
            
            print(f"   ✅ [FR Parser] Successfully validated {len(validated)} requirements")
            return validated
            
        except json.JSONDecodeError as e:
            print(f"   ❌ [FR Parser] JSON decode error: {e}")
            print(f"   📍 [FR Parser] Error at position {e.pos}")
            print(f"   📍 [FR Parser] Around: {json_clean[max(0, e.pos-50):e.pos+50]}")
            return []
        except Exception as e:
            print(f"   ❌ [FR Parser] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            print(f"   📍 [FR Parser] Traceback: {traceback.format_exc()}")
            return []
    
    def _fallback_requirements(self, service_name: str) -> list[Requirement]:
        """Fallback requirements template"""
        return [
            Requirement(
                id="FR-001",
                type=RequirementType.FUNCTIONAL,
                priority=RequirementPriority.HIGH,
                title=f"Implement {service_name} core functionality",
                description=f"The service must implement the core business logic for {service_name}",
                acceptance_criteria=[
                    "Service implements all required business operations",
                    "All operations complete successfully",
                    "Error handling is in place"
                ]
            ),
            Requirement(
                id="FR-002",
                type=RequirementType.FUNCTIONAL,
                priority=RequirementPriority.MEDIUM,
                title=f"Data validation and integrity for {service_name}",
                description="All input data must be validated and sanitized",
                acceptance_criteria=[
                    "Input validation rules are defined",
                    "Invalid data is rejected with clear error messages",
                    "Data integrity is maintained"
                ]
            )
        ]


class APIEndpointsGenerator(dspy.Module):
    """Module for generating API endpoints"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAPIEndpoints)
    
    def forward(self, subdomain: Subdomain, requirements: list[Requirement], progress_callback: Optional[Callable] = None) -> list[APIEndpoint]:
        """Generate REST API endpoints based on requirements"""
        try:
            if progress_callback:
                progress_callback("Preparing API endpoint generation...")
            
            req_summary = "\n".join([
                f"- {req.id}: {req.title}"
                for req in requirements[:5]
            ])
            
            if progress_callback:
                progress_callback("Calling LLM for API endpoints...")
            
            result = self.generate(
                service_name=subdomain.name,
                functional_requirements=req_summary
            )
            
            if progress_callback:
                progress_callback("Parsing API endpoints...")
            
            endpoints = self._parse_endpoints(result.endpoints_json)
            
            if endpoints:
                if progress_callback:
                    progress_callback(f"✅ Generated {len(endpoints)} API endpoints")
                return endpoints
            else:
                if progress_callback:
                    progress_callback("⚠️ Using fallback endpoints")
                return self._fallback_endpoints(subdomain)
            
        except Exception as e:
            print(f"⚠️ API endpoints generation failed: {e}")
            if progress_callback:
                progress_callback(f"⚠️ Using fallback endpoints due to: {str(e)[:100]}")
            return self._fallback_endpoints(subdomain)
    
    def _parse_endpoints(self, json_str: str) -> list[APIEndpoint]:
        """
        Parse endpoints JSON with Pydantic validation
        """
        print(f"   📍 [API Parser] Raw input length: {len(json_str)} chars")
        print(f"   📍 [API Parser] First 200 chars: {json_str[:200]}")
        
        json_clean = extract_json_from_text(json_str)
        
        if not json_clean:
            print("   ❌ [API Parser] No JSON found in text")
            return []
        
        print(f"   📍 [API Parser] Cleaned JSON length: {len(json_clean)} chars")
        print(f"   📍 [API Parser] Cleaned JSON preview: {json_clean[:300]}")
        
        try:
            data = json.loads(json_clean)
            print(f"   ✅ [API Parser] JSON parsed successfully, type: {type(data)}")
            
            # Handle wrapped JSON
            if isinstance(data, dict):
                print(f"   📍 [API Parser] Got dict with keys: {list(data.keys())}")
                for key in ['endpoints', 'api_endpoints', 'apis', 'items', 'data']:
                    if key in data and isinstance(data[key], list):
                        print(f"   ✅ [API Parser] Found endpoints in key '{key}'")
                        data = data[key]
                        break
                else:
                    print(f"   ⚠️ [API Parser] Converting dict to list")
                    data = [data]
            
            if not isinstance(data, list):
                print(f"   ❌ [API Parser] Data is not a list: {type(data)}")
                return []
            
            print(f"   📍 [API Parser] Processing {len(data)} endpoint items")
            
            validated = []
            for idx, ep in enumerate(data[:7]):
                if not isinstance(ep, dict):
                    print(f"   ⚠️ [API Parser] Item {idx} is not a dict: {type(ep)}")
                    continue
                
                print(f"   📍 [API Parser] Item {idx} keys: {list(ep.keys())}")
                
                try:
                    # Validate with Pydantic
                    endpoint = APIEndpoint(
                        method=ep.get("method", "GET").upper(),
                        path=ep.get("path", "/api/v1/resource"),
                        description=str(ep.get("description", ""))[:500],
                        request_schema=ep.get("request_schema"),
                        response_schema=ep.get("response_schema", {}),
                        authentication_required=bool(ep.get("authentication_required", True)),
                        rate_limit=ep.get("rate_limit", "100/min")
                    )
                    
                    validated.append(endpoint)
                    print(f"   ✅ [API Parser] Item {idx} validated: {endpoint.method} {endpoint.path}")
                    
                except (ValidationError, ValueError) as e:
                    print(f"   ⚠️ [API Parser] Item {idx} validation failed: {e}")
                    continue
            
            print(f"   ✅ [API Parser] Successfully validated {len(validated)} endpoints")
            return validated
            
        except json.JSONDecodeError as e:
            print(f"   ❌ [API Parser] JSON decode error: {e}")
            print(f"   📍 [API Parser] Error at position {e.pos}")
            print(f"   📍 [API Parser] Around: {json_clean[max(0, e.pos-50):e.pos+50]}")
            return []
        except Exception as e:
            print(f"   ❌ [API Parser] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            print(f"   📍 [API Parser] Traceback: {traceback.format_exc()}")
            return []
    
    def _fallback_endpoints(self, subdomain: Subdomain) -> list[APIEndpoint]:
        """Fallback endpoints template"""
        return [
            APIEndpoint(
                method="GET",
                path=f"/api/v1/{subdomain.name}/health",
                description="Health check endpoint",
                request_schema=None,
                response_schema={"status": "string", "timestamp": "string"},
                authentication_required=False,
                rate_limit=None
            ),
            APIEndpoint(
                method="POST",
                path=f"/api/v1/{subdomain.name}",
                description=f"Create new {subdomain.name} entity",
                request_schema={"data": "object"},
                response_schema={"id": "string", "status": "string"},
                authentication_required=True,
                rate_limit="100/min"
            ),
            APIEndpoint(
                method="GET",
                path=f"/api/v1/{subdomain.name}/{{id}}",
                description=f"Retrieve {subdomain.name} by ID",
                request_schema=None,
                response_schema={"id": "string", "data": "object"},
                authentication_required=True,
                rate_limit="1000/min"
            )
        ]


class DomainEventsGenerator(dspy.Module):
    """Module for generating domain events"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateDomainEvents)
    
    def forward(self, subdomain: Subdomain, progress_callback: Optional[Callable] = None) -> list[EventDefinition]:
        """Generate domain-driven events"""
        try:
            if progress_callback:
                progress_callback("Preparing domain events generation...")
            
            patterns = ", ".join([p.value for p in subdomain.communication_patterns])
            responsibilities = "\n- " + "\n- ".join(subdomain.responsibilities[:3])
            
            if progress_callback:
                progress_callback("Calling LLM for domain events...")
            
            result = self.generate(
                service_name=subdomain.name,
                responsibilities=responsibilities,
                communication_patterns=patterns or "async_event"
            )
            
            if progress_callback:
                progress_callback("Parsing domain events...")
            
            events = self._parse_events(result.events_json)
            
            if events and progress_callback:
                progress_callback(f"✅ Generated {len(events)} domain events")
            
            return events if events else []
            
        except Exception as e:
            print(f"⚠️ Events generation failed: {e}")
            if progress_callback:
                progress_callback(f"⚠️ No events generated: {str(e)[:100]}")
            return []
    
    def _parse_events(self, json_str: str) -> list[EventDefinition]:
        """
        Parse events JSON with Pydantic validation
        """
        print(f"   📍 [Events Parser] Raw input length: {len(json_str)} chars")
        print(f"   📍 [Events Parser] First 200 chars: {json_str[:200]}")
        
        json_clean = extract_json_from_text(json_str)
        
        if not json_clean:
            print("   ❌ [Events Parser] No JSON found in text")
            return []
        
        print(f"   📍 [Events Parser] Cleaned JSON length: {len(json_clean)} chars")
        print(f"   📍 [Events Parser] Cleaned JSON preview: {json_clean[:300]}")
        
        try:
            data = json.loads(json_clean)
            print(f"   ✅ [Events Parser] JSON parsed successfully, type: {type(data)}")
            
            # Handle wrapped JSON
            if isinstance(data, dict):
                print(f"   📍 [Events Parser] Got dict with keys: {list(data.keys())}")
                for key in ['events', 'domain_events', 'items', 'data']:
                    if key in data and isinstance(data[key], list):
                        print(f"   ✅ [Events Parser] Found events in key '{key}'")
                        data = data[key]
                        break
                else:
                    print(f"   ⚠️ [Events Parser] Converting dict to list")
                    data = [data]
            
            if not isinstance(data, list):
                print(f"   ❌ [Events Parser] Data is not a list: {type(data)}")
                return []
            
            print(f"   📍 [Events Parser] Processing {len(data)} event items")
            
            validated = []
            for idx, ev in enumerate(data[:5]):
                if not isinstance(ev, dict):
                    print(f"   ⚠️ [Events Parser] Item {idx} is not a dict: {type(ev)}")
                    continue
                
                print(f"   📍 [Events Parser] Item {idx} keys: {list(ev.keys())}")
                
                try:
                    # Ensure event_name matches pattern
                    event_name = ev.get("event_name", "DomainEvent")
                    if not event_name.endswith("Event"):
                        event_name += "Event"
                    if event_name[0].islower():
                        event_name = event_name[0].upper() + event_name[1:]
                    
                    # Validate with Pydantic
                    event = EventDefinition(
                        event_name=event_name,
                        event_type=ev.get("event_type", "domain"),
                        payload_schema=ev.get("payload_schema", {}),
                        trigger_conditions=ev.get("trigger_conditions", []),
                        consumers=ev.get("consumers", [])
                    )
                    
                    validated.append(event)
                    print(f"   ✅ [Events Parser] Item {idx} validated: {event.event_name}")
                    
                except (ValidationError, ValueError) as e:
                    print(f"   ⚠️ [Events Parser] Item {idx} validation failed: {e}")
                    continue
            
            print(f"   ✅ [Events Parser] Successfully validated {len(validated)} events")
            return validated
            
        except json.JSONDecodeError as e:
            print(f"   ❌ [Events Parser] JSON decode error: {e}")
            print(f"   📍 [Events Parser] Error at position {e.pos}")
            print(f"   📍 [Events Parser] Around: {json_clean[max(0, e.pos-50):e.pos+50]}")
            return []
        except Exception as e:
            print(f"   ❌ [Events Parser] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            print(f"   📍 [Events Parser] Traceback: {traceback.format_exc()}")
            return []


class NonFunctionalRequirementsGenerator(dspy.Module):
    """Module for generating non-functional requirements (NFRs)"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateNonFunctionalRequirements)
    
    def forward(self, subdomain: Subdomain, progress_callback: Optional[Callable] = None) -> list[Requirement]:
        """Generate NFRs based on service type"""
        try:
            if progress_callback:
                progress_callback("Preparing NFR generation...")
            
            if progress_callback:
                progress_callback("Calling LLM for NFRs...")
            
            result = self.generate(
                service_name=subdomain.name,
                service_type=subdomain.type.value
            )
            
            if progress_callback:
                progress_callback("Parsing NFRs...")
            
            nfrs = self._parse_nfrs(result.nfr_json)
            
            if nfrs:
                if progress_callback:
                    progress_callback(f"✅ Generated {len(nfrs)} NFRs")
                return nfrs
            else:
                if progress_callback:
                    progress_callback("⚠️ Using fallback NFRs")
                return self._fallback_nfrs(subdomain.name)
            
        except Exception as e:
            print(f"⚠️ NFRs generation failed: {e}")
            if progress_callback:
                progress_callback(f"⚠️ Using fallback NFRs: {str(e)[:100]}")
            return self._fallback_nfrs(subdomain.name)
    
    def _parse_nfrs(self, json_str: str) -> list[Requirement]:
        """
        Parse NFRs JSON with Pydantic validation
        """
        print(f"   📍 [NFR Parser] Raw input length: {len(json_str)} chars")
        print(f"   📍 [NFR Parser] First 200 chars: {json_str[:200]}")
        
        json_clean = extract_json_from_text(json_str)
        
        if not json_clean:
            print("   ❌ [NFR Parser] No JSON found in text")
            return []
        
        print(f"   📍 [NFR Parser] Cleaned JSON length: {len(json_clean)} chars")
        print(f"   📍 [NFR Parser] Cleaned JSON preview: {json_clean[:300]}")
        
        try:
            data = json.loads(json_clean)
            print(f"   ✅ [NFR Parser] JSON parsed successfully, type: {type(data)}")
            
            # Handle wrapped JSON
            if isinstance(data, dict):
                print(f"   📍 [NFR Parser] Got dict with keys: {list(data.keys())}")
                
                for key in ['nfr', 'nfrs', 'requirements', 'non_functional_requirements', 'items', 'data']:
                    if key in data and isinstance(data[key], list):
                        print(f"   ✅ [NFR Parser] Found NFRs in key '{key}'")
                        data = data[key]
                        break
                else:
                    print(f"   ⚠️ [NFR Parser] Converting dict to list")
                    data = [data]
            
            if not isinstance(data, list):
                print(f"   ❌ [NFR Parser] Data is not a list after conversion: {type(data)}")
                return []
            
            print(f"   📍 [NFR Parser] Processing {len(data)} NFR items")
            
            validated = []
            for idx, nfr in enumerate(data[:5]):
                if not isinstance(nfr, dict):
                    print(f"   ⚠️ [NFR Parser] Item {idx} is not a dict: {type(nfr)}")
                    continue
                
                print(f"   📍 [NFR Parser] Item {idx} keys: {list(nfr.keys())}")
                
                try:
                    # Validate with Pydantic
                    requirement = Requirement(
                        id=nfr.get("id", f"NFR-{len(validated)+1:03d}"),
                        type=RequirementType.NON_FUNCTIONAL,
                        priority=RequirementPriority(nfr.get("priority", "medium").lower()),
                        title=str(nfr.get("title", ""))[:200],
                        description=str(nfr.get("description", ""))[:1000],
                        acceptance_criteria=nfr.get("acceptance_criteria", [])
                    )
                    
                    if not requirement.title:
                        print(f"   ⚠️ [NFR Parser] Item {idx} missing title, skipping")
                        continue
                    
                    validated.append(requirement)
                    print(f"   ✅ [NFR Parser] Item {idx} validated: {requirement.title}")
                    
                except (ValidationError, ValueError) as e:
                    print(f"   ⚠️ [NFR Parser] Item {idx} validation failed: {e}")
                    continue
            
            print(f"   ✅ [NFR Parser] Successfully validated {len(validated)} NFRs")
            return validated
            
        except json.JSONDecodeError as e:
            print(f"   ❌ [NFR Parser] JSON decode error: {e}")
            print(f"   📍 [NFR Parser] Error at position {e.pos}")
            print(f"   📍 [NFR Parser] Around: {json_clean[max(0, e.pos-50):e.pos+50]}")
            return []
        except Exception as e:
            print(f"   ❌ [NFR Parser] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            print(f"   📍 [NFR Parser] Traceback: {traceback.format_exc()}")
            return []
    
    def _fallback_nfrs(self, service_name: str) -> list[Requirement]:
        """Fallback NFRs template"""
        return [
            Requirement(
                id="NFR-001",
                type=RequirementType.NON_FUNCTIONAL,
                priority=RequirementPriority.HIGH,
                title="Response Time SLA",
                description=f"{service_name} must maintain low latency",
                acceptance_criteria=[
                    "P95 latency < 200ms for read operations",
                    "P99 latency < 500ms for write operations"
                ]
            ),
            Requirement(
                id="NFR-002",
                type=RequirementType.NON_FUNCTIONAL,
                priority=RequirementPriority.HIGH,
                title="Service Availability",
                description="High availability requirement",
                acceptance_criteria=[
                    "99.9% uptime SLA",
                    "Max 43 minutes downtime/month"
                ]
            )
        ]


# ============================================================================
# MAIN PIPELINE - DSPy Modules Orchestration with Streaming Support
# ============================================================================

class SpecificationGeneratorPipeline(dspy.Module):
    """Complete DSPy pipeline for specification generation with progress callbacks"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize LLM engine once
        initialize_llm_engine()
        
        self.req_generator = FunctionalRequirementsGenerator()
        self.api_generator = APIEndpointsGenerator()
        self.events_generator = DomainEventsGenerator()
        self.nfr_generator = NonFunctionalRequirementsGenerator()
    
    def forward(
        self,
        subdomain: Subdomain,
        tech_stack: dict[str, Any],
        global_constraints: dict[str, Any],
        skip_details: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> MicroserviceSpec:
        """
        Generate complete specification for a single subdomain
        
        Args:
            subdomain: The subdomain to generate specs for
            tech_stack: Technical stack preferences
            global_constraints: Global project constraints
            skip_details: If True, skip detailed generation (API, Events, NFRs)
            progress_callback: Optional callback for progress updates
        
        Returns:
            Complete MicroserviceSpec with Pydantic validation
        """
        def _progress(stage: str, status: str = "info", message: str = ""):
            """Internal progress helper"""
            if progress_callback:
                progress_callback(stage, status, message)
        
        _progress("starting", "info", f"Starting generation for {subdomain.name}")
        
        # Step 1: Generate functional requirements
        _progress("requirements", "info", "Starting functional requirements generation")
        requirements = self.req_generator(subdomain, progress_callback=lambda msg: _progress("requirements", "info", msg))
        
        if skip_details:
            _progress("completed", "success", "Completed (minimal mode)")
            return self._build_minimal_spec(subdomain, requirements)
        
        # Step 2: Generate API endpoints
        _progress("api", "info", "Starting API endpoints generation")
        api_endpoints = self.api_generator(subdomain, requirements, progress_callback=lambda msg: _progress("api", "info", msg))
        
        # Step 3: Generate domain events
        _progress("events", "info", "Starting domain events generation")
        events = self.events_generator(subdomain, progress_callback=lambda msg: _progress("events", "info", msg))
        
        # Step 4: Generate NFRs
        _progress("nfr", "info", "Starting NFR generation")
        nfrs = self.nfr_generator(subdomain, progress_callback=lambda msg: _progress("nfr", "info", msg))
        
        # Step 5: Build complete specification
        _progress("assembling", "info", "Assembling final specification")
        spec = self._build_complete_spec(
            subdomain,
            requirements,
            api_endpoints,
            events,
            nfrs,
            tech_stack,
            global_constraints
        )
        
        _progress("completed", "success", f"✅ {subdomain.name} completed successfully")
        return spec
    
    def _build_minimal_spec(self, subdomain: Subdomain, requirements: list[Requirement]) -> MicroserviceSpec:
        """Build minimal specification (requirements only)"""
        return MicroserviceSpec(
            service_name=subdomain.name,
            version="1.0.0",
            description=subdomain.description,
            bounded_context=subdomain.bounded_context,
            functional_requirements=requirements,
            non_functional_requirements=[],
            technology_stack={},
            infrastructure_requirements={},
            monitoring_requirements=MONITORING_REQUIREMENTS
        )
    
    def _build_complete_spec(
        self,
        subdomain: Subdomain,
        requirements: list[Requirement],
        api_endpoints: list[APIEndpoint],
        events: list[EventDefinition],
        nfrs: list[Requirement],
        tech_stack: dict[str, Any],
        global_constraints: dict[str, Any]
    ) -> MicroserviceSpec:
        """Build complete specification with all components"""
        try:
            return MicroserviceSpec(
                service_name=subdomain.name,
                version="1.0.0",
                description=subdomain.description,
                bounded_context=subdomain.bounded_context,
                functional_requirements=requirements,
                non_functional_requirements=nfrs,
                api_endpoints=api_endpoints,
                events_published=[e for e in events if e.event_type != "integration"],
                events_subscribed=[],
                dependencies=subdomain.dependencies or [],
                technology_stack={**DEFAULT_TECH_STACK, **tech_stack},
                infrastructure_requirements=INFRASTRUCTURE_TEMPLATE,
                monitoring_requirements=MONITORING_REQUIREMENTS
            )
        except ValidationError as e:
            print(f"⚠️ Spec validation failed: {e}")
            return self._build_minimal_spec(subdomain, requirements)


# ============================================================================
# MAIN ORCHESTRATOR - High-Level API with SSE Streaming
# ============================================================================

class SpecificationOrchestrator:
    """Main orchestrator for specification generation with SSE streaming support"""
    
    def __init__(self):
        self.pipeline = SpecificationGeneratorPipeline()
    
    async def generate_all_specs_streaming(
        self, 
        architecture_input: ArchitectureInput,
        skip_details: bool = False
    ) -> AsyncGenerator[dict, None]:
        """
        Generate specifications with TRUE SSE streaming.
        
        KEY FIX: Uses asyncio.to_thread to run synchronous pipeline in thread pool
        while allowing async yields to send events immediately.
        """
        microservices = []
        total = len(architecture_input.subdomains)
        
        for idx, subdomain in enumerate(architecture_input.subdomains, 1):
            subdomain_name = subdomain.name
            
            # Event: subdomain started
            yield {
                "event": "progress",
                "data": json.dumps({
                    "subdomain": subdomain_name,
                    "index": idx,
                    "total": total,
                    "step": "started",
                    "message": f"🚀 Starting generation for {subdomain_name}",
                    "timestamp": datetime.utcnow().isoformat()
                })
            }
            
            # ✅ CRITICAL: Ensure event is sent before blocking operation
            await asyncio.sleep(0.01)  # Force event loop to process yield
            
            try:
                # Create a queue for progress updates
                progress_queue = asyncio.Queue()
                
                # Define progress callback that puts messages in queue
                def progress_callback(stage: str, status: str, message: str):
                    """Non-blocking callback that queues progress messages"""
                    try:
                        # Use put_nowait since we're in sync context
                        progress_queue.put_nowait({
                            "subdomain": subdomain_name,
                            "stage": stage,
                            "status": status,
                            "message": message,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    except:
                        pass  # Queue full, skip this update
                
                # ✅ KEY FIX: Run synchronous pipeline in thread pool
                # This allows the async generator to continue yielding events
                pipeline_task = asyncio.create_task(
                    asyncio.to_thread(
                        self.pipeline,
                        subdomain,
                        architecture_input.technical_stack or {},
                        architecture_input.global_constraints or {},
                        skip_details,
                        progress_callback
                    )
                )
                
                # Process progress updates while pipeline runs
                spec = None
                while not pipeline_task.done():
                    try:
                        # Wait for progress update with timeout
                        progress_data = await asyncio.wait_for(
                            progress_queue.get(), 
                            timeout=0.1
                        )
                        
                        # Yield progress event immediately
                        yield {
                            "event": "progress",
                            "data": json.dumps({
                                "subdomain": subdomain_name,
                                "step": progress_data["stage"],
                                "status": progress_data["status"],
                                "message": progress_data["message"],
                                "timestamp": progress_data["timestamp"]
                            })
                        }
                        
                    except asyncio.TimeoutError:
                        # No update available, continue waiting
                        await asyncio.sleep(0.1)
                
                # Get final result
                spec = await pipeline_task
                
                # Process any remaining progress messages
                while not progress_queue.empty():
                    try:
                        progress_data = progress_queue.get_nowait()
                        yield {
                            "event": "progress",
                            "data": json.dumps({
                                "subdomain": subdomain_name,
                                "step": progress_data["stage"],
                                "status": progress_data["status"],
                                "message": progress_data["message"],
                                "timestamp": progress_data["timestamp"]
                            })
                        }
                    except:
                        break
                
                microservices.append(spec)
                
                # Event: microservice completed
                yield {
                    "event": "microservice",
                    "data": json.dumps(spec.model_dump(), default=str)
                }
                
                # Event: subdomain completed
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "step": "completed",
                        "message": f"✅ {subdomain_name} completed successfully",
                        "progress_percent": int((idx / total) * 100),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                }
                
                # ✅ Force event processing
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"❌ Error processing {subdomain_name}: {e}")
                
                # Event: error
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "step": "error",
                        "message": f"❌ Error: {str(e)}",
                        "using_fallback": True,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                }
                
                await asyncio.sleep(0.01)
                
                # Use fallback
                minimal_spec = self._generate_minimal_spec(subdomain)
                microservices.append(minimal_spec)
                
                yield {
                    "event": "microservice",
                    "data": json.dumps(minimal_spec.model_dump(), default=str)
                }
        
        # Final complete event
        try:
            final_output = FunctionalSpecificationOutput(
                project_name=architecture_input.project_name,
                project_description=architecture_input.project_description,
                specification_version="1.0.0",
                microservices=microservices,
                inter_service_communication=self._build_communication_map(architecture_input),
                shared_infrastructure=SHARED_INFRASTRUCTURE
            )
            
            yield {
                "event": "complete",
                "data": json.dumps(final_output.model_dump(), default=str)
            }
        except ValidationError as e:
            print(f"⚠️ Final validation failed: {e}")
            yield {
                "event": "complete",
                "data": json.dumps({
                    "project_name": architecture_input.project_name,
                    "microservices": [m.model_dump() for m in microservices],
                    "validation_error": str(e)
                }, default=str)
            }
    
    def _build_communication_map(self, architecture_input: ArchitectureInput) -> dict[str, list[str]]:
        """Build inter-service communication map"""
        comm_map = {}
        for subdomain in architecture_input.subdomains:
            comm_map[subdomain.name] = subdomain.dependencies or []
        return comm_map
    
    def _generate_minimal_spec(self, subdomain: Subdomain) -> MicroserviceSpec:
        """Minimal fallback spec when everything fails"""
        return MicroserviceSpec(
            service_name=subdomain.name,
            version="1.0.0",
            description=subdomain.description,
            bounded_context=subdomain.bounded_context,
            functional_requirements=[
                Requirement(
                    id="FR-001",
                    type=RequirementType.FUNCTIONAL,
                    priority=RequirementPriority.HIGH,
                    title=f"Implement {subdomain.name} core functionality",
                    description=f"Implement the core business logic for {subdomain.name}",
                    acceptance_criteria=["Service is operational"]
                )
            ],
            non_functional_requirements=[],
            technology_stack={"language": "Python"},
            infrastructure_requirements={},
            monitoring_requirements=["Basic health check"]
        )