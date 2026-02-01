"""
generator_module.py - DSPy-Based Specification Generator
REFACTORED: Removed redundancy, added shared utilities, streaming-only architecture
"""

import json
import re
from datetime import datetime
from typing import Any, Optional, AsyncGenerator

import dspy
from pydantic import ValidationError

from app.models.input_schema import ArchitectureInput, Subdomain
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
    
    This utility function removes redundancy across all generator classes.
    
    Args:
        text: Raw text that may contain JSON
        
    Returns:
        Extracted JSON string or None if no JSON found
    """
    # Try to match JSON inside markdown code blocks
    match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try to match raw JSON array or object
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    
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
    "event_type": "domain|integration|notification",
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
# DSPy MODULES - Reusable Components with Chain-of-Thought
# ============================================================================

class FunctionalRequirementsGenerator(dspy.Module):
    """Module for generating functional requirements with CoT"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateFunctionalRequirements)
    
    def forward(self, subdomain: Subdomain) -> list[dict]:
        """Generate functional requirements with step-by-step reasoning"""
        try:
            responsibilities_str = "\n- " + "\n- ".join(subdomain.responsibilities[:5])
            
            result = self.generate(
                service_name=subdomain.name,
                service_description=subdomain.description,
                responsibilities=responsibilities_str,
                bounded_context=subdomain.bounded_context
            )
            
            requirements = self._parse_and_validate(result.requirements_json, subdomain.name)
            
            if not requirements:
                raise ValueError("Empty requirements from LLM")
            
            return requirements
            
        except Exception as e:
            print(f"⚠️ Requirements generation failed for {subdomain.name}: {e}")
            return self._fallback_requirements(subdomain)
    
    def _parse_and_validate(self, json_str: str, service_name: str) -> list[dict]:
        """Parse JSON with robust validation"""
        json_clean = extract_json_from_text(json_str)
        
        if not json_clean:
            return []
        
        try:
            data = json.loads(json_clean)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return []
        
        if not isinstance(data, list):
            data = [data]
        
        validated = []
        for idx, req in enumerate(data[:5]):
            if not isinstance(req, dict):
                continue
            
            normalized = {
                "id": req.get("id", f"FR-{idx+1:03d}"),
                "type": "functional",
                "priority": self._normalize_priority(req.get("priority", "medium")),
                "title": str(req.get("title", f"{service_name} requirement {idx+1}"))[:200],
                "description": str(req.get("description", "To be defined"))[:1000],
                "acceptance_criteria": self._normalize_criteria(req.get("acceptance_criteria", [])),
                "related_requirements": req.get("related_requirements", [])
            }
            validated.append(normalized)
        
        return validated
    
    def _normalize_priority(self, priority: str) -> str:
        """Normalize priority values"""
        p = str(priority).lower()
        if "critical" in p:
            return "critical"
        elif "high" in p:
            return "high"
        elif "low" in p:
            return "low"
        return "medium"
    
    def _normalize_criteria(self, criteria: Any) -> list[str]:
        """Normalize acceptance criteria"""
        if isinstance(criteria, list):
            return [str(c)[:200] for c in criteria if c][:5]
        elif isinstance(criteria, str):
            return [criteria[:200]]
        return ["Acceptance criteria to be defined"]
    
    def _fallback_requirements(self, subdomain: Subdomain) -> list[dict]:
        """Fallback template when generation fails"""
        return [
            {
                "id": "FR-001",
                "type": "functional",
                "priority": "high",
                "title": f"Core {subdomain.name} functionality",
                "description": f"Implement core business logic for {subdomain.name} based on: {', '.join(subdomain.responsibilities[:3])}",
                "acceptance_criteria": [
                    "Service is operational and responds to health checks",
                    "Core domain logic is implemented",
                    "Basic CRUD operations are functional"
                ],
                "related_requirements": []
            },
            {
                "id": "FR-002",
                "type": "functional",
                "priority": "medium",
                "title": f"Data persistence for {subdomain.name}",
                "description": f"Implement data storage and retrieval mechanisms",
                "acceptance_criteria": [
                    "Data is persisted correctly",
                    "Data integrity is maintained",
                    "Query performance meets SLA"
                ],
                "related_requirements": ["FR-001"]
            }
        ]


class APIEndpointsGenerator(dspy.Module):
    """Module for generating API endpoints"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAPIEndpoints)
    
    def forward(self, subdomain: Subdomain, requirements: list[dict]) -> list[dict]:
        """Generate REST API endpoints based on requirements"""
        try:
            req_summary = "\n".join([
                f"- {req['id']}: {req['title']}"
                for req in requirements[:5]
            ])
            
            result = self.generate(
                service_name=subdomain.name,
                functional_requirements=req_summary
            )
            
            endpoints = self._parse_endpoints(result.endpoints_json)
            return endpoints if endpoints else self._fallback_endpoints(subdomain)
            
        except Exception as e:
            print(f"⚠️ API endpoints generation failed: {e}")
            return self._fallback_endpoints(subdomain)
    
    def _parse_endpoints(self, json_str: str) -> list[dict]:
        """Parse endpoints JSON"""
        json_clean = extract_json_from_text(json_str)
        if not json_clean:
            return []
        
        try:
            data = json.loads(json_clean)
            if not isinstance(data, list):
                data = [data]
            
            validated = []
            for ep in data[:7]:
                if not isinstance(ep, dict):
                    continue
                
                validated.append({
                    "method": ep.get("method", "GET").upper(),
                    "path": ep.get("path", "/api/v1/resource"),
                    "description": str(ep.get("description", ""))[:500],
                    "request_schema": ep.get("request_schema"),
                    "response_schema": ep.get("response_schema", {}),
                    "authentication_required": bool(ep.get("authentication_required", True)),
                    "rate_limit": ep.get("rate_limit", "100/min")
                })
            
            return validated
        except:
            return []
    
    def _fallback_endpoints(self, subdomain: Subdomain) -> list[dict]:
        """Fallback endpoints template"""
        return [
            {
                "method": "GET",
                "path": f"/api/v1/{subdomain.name}/health",
                "description": "Health check endpoint",
                "request_schema": None,
                "response_schema": {"status": "string", "timestamp": "string"},
                "authentication_required": False,
                "rate_limit": None
            },
            {
                "method": "POST",
                "path": f"/api/v1/{subdomain.name}",
                "description": f"Create new {subdomain.name} entity",
                "request_schema": {"data": "object"},
                "response_schema": {"id": "string", "status": "string"},
                "authentication_required": True,
                "rate_limit": "100/min"
            },
            {
                "method": "GET",
                "path": f"/api/v1/{subdomain.name}/{{id}}",
                "description": f"Retrieve {subdomain.name} by ID",
                "request_schema": None,
                "response_schema": {"id": "string", "data": "object"},
                "authentication_required": True,
                "rate_limit": "1000/min"
            }
        ]


class DomainEventsGenerator(dspy.Module):
    """Module for generating domain events"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateDomainEvents)
    
    def forward(self, subdomain: Subdomain) -> list[dict]:
        """Generate domain-driven events"""
        try:
            patterns = ", ".join([p.value for p in subdomain.communication_patterns])
            responsibilities = "\n- " + "\n- ".join(subdomain.responsibilities[:3])
            
            result = self.generate(
                service_name=subdomain.name,
                responsibilities=responsibilities,
                communication_patterns=patterns or "async_event"
            )
            
            events = self._parse_events(result.events_json)
            return events if events else []
            
        except Exception as e:
            print(f"⚠️ Events generation failed: {e}")
            return []
    
    def _parse_events(self, json_str: str) -> list[dict]:
        """Parse events JSON"""
        json_clean = extract_json_from_text(json_str)
        if not json_clean:
            return []
        
        try:
            data = json.loads(json_clean)
            if not isinstance(data, list):
                data = [data]
            
            validated = []
            for ev in data[:5]:
                if not isinstance(ev, dict):
                    continue
                
                validated.append({
                    "event_name": ev.get("event_name", "DomainEvent"),
                    "event_type": ev.get("event_type", "domain"),
                    "payload_schema": ev.get("payload_schema", {}),
                    "trigger_conditions": ev.get("trigger_conditions", []),
                    "consumers": ev.get("consumers", [])
                })
            
            return validated
        except:
            return []


class NonFunctionalRequirementsGenerator(dspy.Module):
    """Module for generating NFRs"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateNonFunctionalRequirements)
    
    def forward(self, subdomain: Subdomain) -> list[dict]:
        """Generate non-functional requirements"""
        try:
            result = self.generate(
                service_name=subdomain.name,
                service_type=subdomain.type.value
            )
            
            nfrs = self._parse_nfrs(result.nfr_json)
            return nfrs if nfrs else self._fallback_nfrs(subdomain.name)
            
        except Exception as e:
            print(f"⚠️ NFRs generation failed: {e}")
            return self._fallback_nfrs(subdomain.name)
    
    def _parse_nfrs(self, json_str: str) -> list[dict]:
        """Parse NFRs JSON"""
        json_clean = extract_json_from_text(json_str)
        if not json_clean:
            return []
        
        try:
            data = json.loads(json_clean)
            if not isinstance(data, list):
                data = [data]
            
            validated = []
            for nfr in data[:5]:
                if not isinstance(nfr, dict):
                    continue
                
                validated.append({
                    "id": nfr.get("id", f"NFR-{len(validated)+1:03d}"),
                    "type": "non_functional",
                    "priority": nfr.get("priority", "medium"),
                    "title": str(nfr.get("title", ""))[:200],
                    "description": str(nfr.get("description", ""))[:1000],
                    "acceptance_criteria": nfr.get("acceptance_criteria", []),
                    "related_requirements": []
                })
            
            return validated
        except:
            return []
    
    def _fallback_nfrs(self, service_name: str) -> list[dict]:
        """Fallback NFRs template"""
        return [
            {
                "id": "NFR-001",
                "type": "non_functional",
                "priority": "high",
                "title": "Response Time SLA",
                "description": f"{service_name} must maintain low latency",
                "acceptance_criteria": [
                    "P95 latency < 200ms for read operations",
                    "P99 latency < 500ms for write operations"
                ],
                "related_requirements": []
            },
            {
                "id": "NFR-002",
                "type": "non_functional",
                "priority": "high",
                "title": "Service Availability",
                "description": "High availability requirement",
                "acceptance_criteria": [
                    "99.9% uptime SLA",
                    "Max 43 minutes downtime/month"
                ],
                "related_requirements": []
            }
        ]


# ============================================================================
# MAIN PIPELINE - DSPy Modules Orchestration
# ============================================================================

class SpecificationGeneratorPipeline(dspy.Module):
    """Complete DSPy pipeline for specification generation"""
    
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
        technical_stack: dict[str, str],
        global_constraints: dict[str, str],
        skip_details: bool = False
    ) -> dict[str, Any]:
        """Execute complete pipeline for a subdomain"""
        service_name = subdomain.name
        print(f"\n{'='*60}")
        print(f"🔄 Processing: {service_name} (Fast Mode: {skip_details})")
        print(f"{'='*60}")
        
        # 1. Functional Requirements - Sempre generati
        print(f"📋 [1/4] Generating functional requirements...")
        functional_reqs = self.req_generator(subdomain)
        print(f"   ✅ Generated {len(functional_reqs)} requirements")
        
        if skip_details:
            # Salta le chiamate LLM pesanti per i test
            print(f"⏩ [SKIP] Skipping detailed generators as requested...")
            api_endpoints = []
            events_published = []
            nfr_reqs = []
        else:
            # Esecuzione standard completa
            print(f"🔌 [2/4] Generating API endpoints...")
            api_endpoints = self.api_generator(subdomain, functional_reqs)
            print(f"   ✅ Generated {len(api_endpoints)} endpoints")
            
            print(f"📡 [3/4] Generating domain events...")
            events_published = self.events_generator(subdomain)
            print(f"   ✅ Generated {len(events_published)} events")
            
            print(f"⚡ [4/4] Generating NFRs...")
            nfr_reqs = self.nfr_generator(subdomain)
            print(f"   ✅ Generated {len(nfr_reqs)} NFRs")
        
        return {
            "service_name": service_name,
            "version": "1.0.0",
            "description": subdomain.description,
            "bounded_context": subdomain.bounded_context,
            "functional_requirements": functional_reqs,
            "non_functional_requirements": nfr_reqs,
            "events_published": events_published,
            "events_subscribed": [],
            "api_endpoints": api_endpoints,
            "message_queues": [],
            "dependencies": self._convert_dependencies(subdomain),
            "technology_stack": technical_stack or DEFAULT_TECH_STACK,
            "infrastructure_requirements": INFRASTRUCTURE_TEMPLATE,
            "monitoring_requirements": MONITORING_REQUIREMENTS,
            "generated_at": datetime.utcnow(),
            "generated_by": "agent3-spec-generator-dspy"
        }
    
    def _convert_dependencies(self, subdomain: Subdomain) -> list[dict]:
        """Convert dependencies from input schema"""
        return [
            {
                "service_name": dep,
                "dependency_type": "service",
                "communication_method": "rest",
                "criticality": "high",
                "fallback_strategy": "circuit_breaker"
            }
            for dep in (subdomain.dependencies or [])
        ]


# ============================================================================
# ORCHESTRATOR - Streaming-Only Entry Point
# ============================================================================

class SpecificationOrchestrator:
    """Orchestrator with SSE streaming support (streaming-only architecture)"""
    
    def __init__(self):
        self.pipeline = SpecificationGeneratorPipeline()
    
    async def generate_all_specs_streaming(
        self, 
        architecture_input: ArchitectureInput,
        skip_details: bool = False
    ) -> AsyncGenerator[dict, None]:
        """
        Generate specifications with SSE streaming.
        Yields SSE events for each step of the process.
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
                    "message": f"Starting generation for {subdomain_name}"
                })
            }
            
            try:
                # Event: requirements generation
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "step": "requirements",
                        "message": "Generating functional requirements..."
                    })
                }
                
                # Event: api generation
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "step": "api",
                        "message": "Generating API endpoints..."
                    })
                }
                
                # Event: events generation
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "step": "events",
                        "message": "Generating domain events..."
                    })
                }
                
                # Event: nfr generation
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "step": "nfr",
                        "message": "Generating non-functional requirements..."
                    })
                }
                
                # Execute pipeline
                spec = self.pipeline(
                    subdomain=subdomain,
                    technical_stack=architecture_input.technical_stack or {},
                    global_constraints=architecture_input.global_constraints or {},
                    skip_details=skip_details
                )
                
                microservices.append(spec)
                
                # Event: microservice completed
                yield {
                    "event": "microservice",
                    "data": json.dumps(spec, default=str)
                }
                
                # Event: subdomain completed
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "step": "completed",
                        "message": f"✅ {subdomain_name} completed successfully",
                        "progress_percent": int((idx / total) * 100)
                    })
                }
                
            except Exception as e:
                # Event: error
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "step": "error",
                        "message": f"Error: {str(e)}",
                        "using_fallback": True
                    })
                }
                
                # Use fallback
                minimal_spec = self._generate_minimal_spec(subdomain)
                microservices.append(minimal_spec)
                
                yield {
                    "event": "microservice",
                    "data": json.dumps(minimal_spec, default=str)
                }
        
        # Final complete event
        final_output = {
            "project_name": architecture_input.project_name,
            "project_description": architecture_input.project_description,
            "specification_version": "1.0.0",
            "microservices": microservices,
            "inter_service_communication": self._build_communication_map(architecture_input),
            "shared_infrastructure": SHARED_INFRASTRUCTURE,
            "generated_at": datetime.utcnow(),
            "agent_version": "1.0.0"
        }
        
        yield {
            "event": "complete",
            "data": json.dumps(final_output, default=str)
        }
    
    def _build_communication_map(self, architecture_input: ArchitectureInput) -> dict[str, list[str]]:
        """Build inter-service communication map"""
        comm_map = {}
        for subdomain in architecture_input.subdomains:
            comm_map[subdomain.name] = subdomain.dependencies or []
        return comm_map
    
    def _generate_minimal_spec(self, subdomain: Subdomain) -> dict:
        """Minimal fallback spec when everything fails"""
        return {
            "service_name": subdomain.name,
            "version": "1.0.0",
            "description": subdomain.description,
            "bounded_context": subdomain.bounded_context,
            "functional_requirements": [
                {
                    "id": "FR-001",
                    "type": "functional",
                    "priority": "high",
                    "title": f"Implement {subdomain.name} core functionality",
                    "description": f"Implement the core business logic for {subdomain.name}",
                    "acceptance_criteria": ["Service is operational"],
                    "related_requirements": []
                }
            ],
            "non_functional_requirements": [],
            "events_published": [],
            "events_subscribed": [],
            "api_endpoints": [],
            "dependencies": [],
            "technology_stack": {},
            "infrastructure_requirements": {},
            "monitoring_requirements": ["Basic health check"],
            "generated_at": datetime.utcnow(),
            "generated_by": "agent3-fallback"
        }