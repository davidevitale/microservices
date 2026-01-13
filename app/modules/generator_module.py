"""
generator_module.py - SOLUZIONE: HTTP Diretto + Timeout Reali
"""

import json
import re
from datetime import datetime
from typing import Any, Optional
import httpx
from app.models.input_schema import ArchitectureInput, Subdomain
from app.core.config import settings


class OllamaClient:
    """Client HTTP diretto per Ollama con timeout controllati"""
    
    def __init__(self, base_url: str, model: str, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
    
    def generate(self, prompt: str, system: str = "") -> str:
        """
        Chiamata HTTP diretta con timeout reale
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": system,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Più deterministico
                            "num_predict": 800,  # Ridotto da 1000
                            "num_ctx": 2048,     # Context window ridotto
                            "top_p": 0.9,        # Più focalizzato
                            "top_k": 40,         # Riduce opzioni
                        }
                    }
                )
                response.raise_for_status()
                return response.json()["response"]
        except httpx.TimeoutException:
            raise TimeoutError(f"Ollama timeout after {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")


class SpecificationGeneratorPipeline:
    """
    Pipeline semplificata senza DSPy
    """

    def __init__(self):
        self.ollama = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=300  # 5 minuti per chiamata LLM
        )

    def forward(
        self,
        subdomain: Subdomain,
        technical_stack: dict[str, str],
        global_constraints: dict[str, str],
    ) -> dict[str, Any]:
        """
        Generate specification con chiamate HTTP dirette
        """
        
        service_name = subdomain.name
        print(f"[{service_name}] Starting generation...")

        # STEP 1: Functional Requirements (CRITICO)
        functional_reqs = self._generate_requirements(subdomain)
        
        # STEP 2: API Endpoints (OPZIONALE)
        api_endpoints = self._generate_endpoints(subdomain, functional_reqs)
        
        # STEP 3: Events (OPZIONALE)
        events_published = self._generate_events(subdomain)

        # ASSEMBLY FINALE
        return {
            "service_name": service_name,
            "version": "1.0.0",
            "description": subdomain.description,
            "bounded_context": subdomain.bounded_context,
            "functional_requirements": functional_reqs,
            "non_functional_requirements": self._generate_nfr_template(service_name),
            "events_published": events_published,
            "events_subscribed": [],
            "api_endpoints": api_endpoints,
            "dependencies": self._convert_dependencies(subdomain),
            "technology_stack": technical_stack or {"language": "Python", "framework": "FastAPI"},
            "infrastructure_requirements": self._generate_infra_template(),
            "monitoring_requirements": [
                "Prometheus metrics endpoint /metrics",
                "Health check endpoint /health",
                "Structured JSON logging",
            ],
            "generated_at": datetime.utcnow(),
            "generated_by": "agent3-spec-generator",
        }

    def _generate_requirements(self, subdomain: Subdomain) -> list[dict]:
        """
        Genera functional requirements con prompt semplificato
        """
        prompt = f"""Generate 3-5 functional requirements for this microservice:

Service: {subdomain.name}
Description: {subdomain.description}
Responsibilities: {', '.join(subdomain.responsibilities[:3])}

Return ONLY a JSON array with this EXACT format (no markdown, no explanation):
[
  {{
    "id": "FR-001",
    "title": "Short title",
    "description": "Detailed description",
    "acceptance_criteria": ["criterion 1", "criterion 2"]
  }}
]"""

        try:
            response = self.ollama.generate(
                prompt=prompt,
                system="You are a technical specification generator. Output ONLY valid JSON."
            )
            return self._parse_requirements(response, subdomain.name)
        except Exception as e:
            print(f"[{subdomain.name}] Requirements generation failed: {e}")
            return self._generate_fallback_requirements(subdomain)

    def _generate_endpoints(self, subdomain: Subdomain, requirements: list[dict]) -> list[dict]:
        """
        Genera API endpoints basati sui requirements
        """
        req_titles = ", ".join([r["title"] for r in requirements[:3]])
        
        prompt = f"""Generate 3-5 REST API endpoints for this service:

Service: {subdomain.name}
Requirements: {req_titles}

Return ONLY a JSON array with this format:
[
  {{
    "method": "POST",
    "path": "/api/v1/resource",
    "description": "Endpoint description",
    "response_schema": {{"id": "string", "status": "string"}}
  }}
]"""

        try:
            response = self.ollama.generate(prompt, system="Output ONLY valid JSON.")
            return self._parse_json_array(response, max_items=5)
        except Exception as e:
            print(f"[{subdomain.name}] Endpoints generation failed: {e}")
            return self._generate_fallback_endpoints(subdomain)

    def _generate_events(self, subdomain: Subdomain) -> list[dict]:
        """
        Genera event definitions
        """
        prompt = f"""Generate 2-3 domain events for this service:

Service: {subdomain.name}
Responsibilities: {', '.join(subdomain.responsibilities[:2])}

Return ONLY a JSON array:
[
  {{
    "event_name": "EntityCreatedEvent",
    "event_type": "domain",
    "payload_schema": {{"entity_id": "string", "timestamp": "string"}},
    "trigger_conditions": ["condition 1"]
  }}
]"""

        try:
            response = self.ollama.generate(prompt, system="Output ONLY valid JSON.")
            return self._parse_json_array(response, max_items=3)
        except Exception as e:
            print(f"[{subdomain.name}] Events generation failed: {e}")
            return []

    # === PARSING ROBUSTO ===
    
    def _parse_requirements(self, raw_text: str, service_name: str) -> list[dict]:
        """
        Parsing ultra-robusto per requirements
        """
        # Step 1: Estrai JSON
        json_data = self._extract_json(raw_text)
        if not json_data:
            return self._generate_fallback_requirements(None)
        
        # Step 2: Valida struttura
        if not isinstance(json_data, list):
            json_data = [json_data]
        
        # Step 3: Normalizza ogni requirement
        requirements = []
        for idx, item in enumerate(json_data[:5]):  # Max 5
            if not isinstance(item, dict):
                continue
            
            req = {
                "id": item.get("id", f"FR-{idx+1:03d}"),
                "type": "functional",
                "priority": self._normalize_priority(item.get("priority", "medium")),
                "title": str(item.get("title", f"Requirement {idx+1}"))[:200],
                "description": str(item.get("description", ""))[:1000],
                "acceptance_criteria": self._normalize_criteria(item.get("acceptance_criteria", [])),
                "related_requirements": [],
            }
            requirements.append(req)
        
        return requirements if requirements else self._generate_fallback_requirements(None)

    def _parse_json_array(self, raw_text: str, max_items: int = 10) -> list[dict]:
        """
        Parsing generico per array JSON
        """
        json_data = self._extract_json(raw_text)
        if not json_data:
            return []
        
        if not isinstance(json_data, list):
            json_data = [json_data]
        
        return [item for item in json_data[:max_items] if isinstance(item, dict)]

    def _extract_json(self, text: str) -> Optional[Any]:
        """
        Estrai JSON da testo sporco (con markdown, spiegazioni, etc)
        """
        # Metodo 1: Trova blocco ```json
        json_block = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', text, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except:
                pass
        
        # Metodo 2: Trova primo array/object
        json_match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Metodo 3: Prova tutto il testo
        try:
            return json.loads(text.strip())
        except:
            return None

    def _normalize_priority(self, priority: str) -> str:
        """Normalizza priority values"""
        priority = str(priority).lower()
        if "critical" in priority or "high" in priority:
            return "high"
        elif "low" in priority:
            return "low"
        return "medium"

    def _normalize_criteria(self, criteria: Any) -> list[str]:
        """Normalizza acceptance criteria"""
        if isinstance(criteria, list):
            return [str(c)[:200] for c in criteria if c][:5]
        elif isinstance(criteria, str):
            return [criteria[:200]]
        return ["Acceptance criteria to be defined"]

    # === FALLBACKS ===
    
    def _generate_fallback_requirements(self, subdomain: Optional[Subdomain]) -> list[dict]:
        """Requirements template quando LLM fallisce"""
        name = subdomain.name if subdomain else "service"
        return [
            {
                "id": "FR-001",
                "type": "functional",
                "priority": "high",
                "title": f"Core {name} functionality",
                "description": f"Implement core business logic for {name} service",
                "acceptance_criteria": [
                    "Service responds to health checks",
                    "Core API endpoints are functional",
                    "Data validation is implemented",
                ],
                "related_requirements": [],
            },
            {
                "id": "FR-002",
                "type": "functional",
                "priority": "medium",
                "title": f"Data persistence for {name}",
                "description": f"Implement data storage and retrieval for {name}",
                "acceptance_criteria": [
                    "Data is persisted correctly",
                    "CRUD operations are functional",
                ],
                "related_requirements": ["FR-001"],
            }
        ]

    def _generate_fallback_endpoints(self, subdomain: Subdomain) -> list[dict]:
        """Endpoints template"""
        return [
            {
                "method": "GET",
                "path": f"/api/v1/{subdomain.name}/health",
                "description": "Health check endpoint",
                "request_schema": None,
                "response_schema": {"status": "string", "timestamp": "string"},
                "authentication_required": False,
                "rate_limit": None,
            },
            {
                "method": "POST",
                "path": f"/api/v1/{subdomain.name}",
                "description": f"Create new {subdomain.name} entity",
                "request_schema": {"data": "object"},
                "response_schema": {"id": "string", "status": "string", "created_at": "string"},
                "authentication_required": True,
                "rate_limit": "100/min",
            },
            {
                "method": "GET",
                "path": f"/api/v1/{subdomain.name}/{{id}}",
                "description": f"Retrieve {subdomain.name} by ID",
                "request_schema": None,
                "response_schema": {"id": "string", "data": "object"},
                "authentication_required": True,
                "rate_limit": "1000/min",
            }
        ]

    def _generate_nfr_template(self, service_name: str) -> list[dict]:
        """Non-functional requirements template"""
        return [
            {
                "id": "NFR-001",
                "type": "non_functional",
                "priority": "high",
                "title": "Response Time SLA",
                "description": f"{service_name} must maintain low latency for all operations",
                "acceptance_criteria": [
                    "P95 latency < 200ms for read operations",
                    "P99 latency < 500ms for all operations"
                ],
                "related_requirements": [],
            },
            {
                "id": "NFR-002",
                "type": "non_functional",
                "priority": "high",
                "title": "Service Availability",
                "description": "Service must maintain high availability",
                "acceptance_criteria": [
                    "99.9% uptime SLA",
                    "Max unplanned downtime: 43 minutes/month"
                ],
                "related_requirements": [],
            },
            {
                "id": "NFR-003",
                "type": "non_functional",
                "priority": "medium",
                "title": "Scalability",
                "description": "Service must scale horizontally",
                "acceptance_criteria": [
                    "Support 10x traffic increase with horizontal scaling",
                    "Stateless service design"
                ],
                "related_requirements": [],
            }
        ]

    def _generate_infra_template(self) -> dict[str, Any]:
        """Infrastructure requirements template"""
        return {
            "database": {
                "type": "PostgreSQL",
                "version": "15+",
                "replicas": 2,
                "backup_strategy": "daily"
            },
            "cache": {
                "type": "Redis",
                "version": "7+",
                "ttl": "1h",
                "persistence": "AOF"
            },
            "message_queue": {
                "type": "RabbitMQ",
                "version": "3.12+",
                "durable": True,
                "prefetch_count": 10
            }
        }

    def _convert_dependencies(self, subdomain: Subdomain) -> list[dict]:
        """Convert dependencies from input schema"""
        return [
            {
                "service_name": dep,
                "dependency_type": "service",
                "communication_method": "rest",
                "criticality": "high",
                "fallback_strategy": "circuit_breaker",
            }
            for dep in (subdomain.dependencies or [])
        ]


class SpecificationOrchestrator:
    """Orchestrator con error handling robusto"""

    def __init__(self):
        self.pipeline = SpecificationGeneratorPipeline()

    def generate_all_specs(self, architecture_input: ArchitectureInput) -> dict[str, Any]:
        """
        Generate specs con progress tracking e fallback garantito
        """
        microservices = []
        total = len(architecture_input.subdomains)

        for idx, subdomain in enumerate(architecture_input.subdomains, 1):
            print(f"\n{'='*60}")
            print(f"Processing {idx}/{total}: {subdomain.name}")
            print(f"{'='*60}")
            
            try:
                spec = self.pipeline.forward(
                    subdomain=subdomain,
                    technical_stack=architecture_input.technical_stack or {},
                    global_constraints=architecture_input.global_constraints or {},
                )
                microservices.append(spec)
                print(f"✅ {subdomain.name} completed successfully")
                
            except Exception as e:
                print(f"❌ {subdomain.name} failed: {e}")
                # Anche in caso di errore, generiamo spec minimale
                microservices.append(self._generate_minimal_spec(subdomain))
                print(f"⚠️  Using fallback spec for {subdomain.name}")

        return {
            "project_name": architecture_input.project_name,
            "project_description": architecture_input.project_description,
            "specification_version": "1.0.0",
            "microservices": microservices,
            "inter_service_communication": self._build_communication_map(architecture_input),
            "shared_infrastructure": {
                "api_gateway": {"type": "Kong", "version": "3.x"},
                "message_broker": {"type": "RabbitMQ", "version": "3.12"},
                "service_mesh": {"type": "Istio", "version": "1.20"},
                "monitoring": {"type": "Prometheus + Grafana"},
                "logging": {"type": "ELK Stack"},
                "tracing": {"type": "Jaeger"}
            },
            "generated_at": datetime.utcnow(),
            "agent_version": "1.0.0",
        }

    def _build_communication_map(self, architecture_input: ArchitectureInput) -> dict[str, list[str]]:
        """Build service communication map"""
        comm_map = {}
        for subdomain in architecture_input.subdomains:
            comm_map[subdomain.name] = subdomain.dependencies or []
        return comm_map

    def _generate_minimal_spec(self, subdomain: Subdomain) -> dict:
        """Spec minimale garantito quando tutto fallisce"""
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
                    "related_requirements": [],
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
            "generated_by": "agent3-fallback",
        }