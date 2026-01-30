"""
generator_module.py - DSPy-Based Specification Generator
COMPLETELY REFACTORED: 5-Phase NFR-First Architecture

PHASE-BASED PIPELINE:
Phase 0: Non-Functional Requirements Analysis (NFR Constraints Extraction)
Phase 1: Functional Requirements Generation (Constraint-Guided)
Phase 2: API Endpoints Generation (FR + NFR Aware)
Phase 3: Domain Events Generation (Scalability + Decoupling)
Phase 4: Correlation Validation (Coherence Check)

Key Principle: NFRs are CONSTRAINTS that guide all other phases, not outputs.
"""

import json
import re
from datetime import datetime
from typing import Any, Optional, AsyncGenerator, Dict, List

import dspy
from pydantic import ValidationError

from app.models.input_schema import ArchitectureInput, Subdomain, SubdomainType
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

# NFR Baselines by Service Type
NFR_BASELINES = {
    "core": {
        "latency_p95_ms": 200,
        "latency_p99_ms": 500,
        "availability_percent": 99.9,
        "requests_per_second": 1000,
        "error_rate_percent": 0.1
    },
    "supporting": {
        "latency_p95_ms": 500,
        "latency_p99_ms": 1000,
        "availability_percent": 99.5,
        "requests_per_second": 500,
        "error_rate_percent": 0.5
    },
    "generic": {
        "latency_p95_ms": 1000,
        "latency_p99_ms": 2000,
        "availability_percent": 99.0,
        "requests_per_second": 200,
        "error_rate_percent": 1.0
    }
}


# ============================================================================
# SHARED UTILITY FUNCTIONS - DRY Compliance
# ============================================================================

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON from text that may contain markdown code blocks or other noise.
    
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


def serialize_context(data: Any) -> str:
    """
    Serialize context data to JSON string for passing to next phase.
    
    Args:
        data: Python object to serialize
        
    Returns:
        JSON string representation
    """
    return json.dumps(data, indent=2, default=str)


# ============================================================================
# PHASE 0: NON-FUNCTIONAL REQUIREMENTS ANALYSIS
# ============================================================================

class AnalyzeNFRConstraints(dspy.Signature):
    """Phase 0: Extract NFR constraints that will guide all subsequent phases"""
    
    service_name: str = dspy.InputField(desc="Service name")
    service_type: str = dspy.InputField(desc="Service type: core|supporting|generic")
    service_description: str = dspy.InputField(desc="Service description")
    responsibilities: str = dspy.InputField(desc="Service responsibilities")
    global_constraints: str = dspy.InputField(desc="Global project constraints (JSON)")
    technical_stack: str = dspy.InputField(desc="Technical stack capabilities (JSON)")
    
    nfr_constraints_json: str = dspy.OutputField(
        desc="""JSON array of NFR constraints that will guide all subsequent phases. EXACT format:
[
  {
    "id": "NFR-001",
    "category": "performance|availability|security|scalability|reliability",
    "priority": "critical|high|medium",
    "constraint": "Specific constraint (e.g., P95 latency < 200ms)",
    "acceptance_criteria": ["Measurable criterion 1", "Measurable criterion 2"],
    "implications": {
      "for_fr": ["How this affects functional requirements design"],
      "for_api": ["How this affects API endpoint design"],
      "for_events": ["How this affects event-driven architecture"]
    }
  }
]

Guidelines based on service type:
- CORE services: P95 < 200ms, 99.9% availability, critical security
- SUPPORTING services: P95 < 500ms, 99.5% availability, high security  
- GENERIC services: P95 < 1s, 99% availability, standard security

Generate 3-5 NFR constraints. Include at least: performance, availability, scalability.
ONLY valid JSON, NO explanations."""
    )


class NonFunctionalRequirementsAnalyzer(dspy.Module):
    """Phase 0: Analyze and extract NFR constraints"""
    
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(AnalyzeNFRConstraints)
    
    def forward(
        self,
        subdomain: Subdomain,
        global_constraints: Dict[str, str],
        technical_stack: Dict[str, str]
    ) -> List[Dict]:
        """
        Extract NFR constraints that will guide all subsequent phases.
        
        Args:
            subdomain: Subdomain definition
            global_constraints: Global project constraints
            technical_stack: Technical stack
            
        Returns:
            List of NFR constraint dictionaries
        """
        try:
            response = self.analyze(
                service_name=subdomain.name,
                service_type=subdomain.type.value,
                service_description=subdomain.description,
                responsibilities="\n".join(f"- {r}" for r in subdomain.responsibilities),
                global_constraints=serialize_context(global_constraints),
                technical_stack=serialize_context(technical_stack)
            )
            
            json_str = extract_json_from_text(response.nfr_constraints_json)
            nfr_constraints, _ = json.JSONDecoder().raw_decode(json_str)
            
            if not isinstance(nfr_constraints, list):
                nfr_constraints = [nfr_constraints]
            
            # Validate structure
            for nfr in nfr_constraints:
                if "id" not in nfr or "category" not in nfr:
                    raise ValueError("Invalid NFR constraint structure")
            
            return nfr_constraints
            
        except Exception as e:
            print(f"⚠️  Phase 0 (NFR Analysis) error: {e}")
            # Fallback: Generate baseline NFR constraints based on service type
            return self._generate_baseline_nfr_constraints(subdomain)
    
    def _generate_baseline_nfr_constraints(self, subdomain: Subdomain) -> List[Dict]:
        """Generate baseline NFR constraints based on service type"""
        baseline = NFR_BASELINES[subdomain.type.value]
        
        return [
            {
                "id": "NFR-001",
                "category": "performance",
                "priority": "critical" if subdomain.type == SubdomainType.CORE else "high",
                "constraint": f"P95 latency < {baseline['latency_p95_ms']}ms",
                "acceptance_criteria": [
                    f"95% of requests complete in < {baseline['latency_p95_ms']}ms",
                    f"99% of requests complete in < {baseline['latency_p99_ms']}ms"
                ],
                "implications": {
                    "for_fr": ["Decompose into atomic operations", "Each FR must be completable within latency budget"],
                    "for_api": [f"Endpoint timeout < {int(baseline['latency_p95_ms'] * 0.75)}ms", "Use async endpoints for long operations"],
                    "for_events": ["Use async processing for non-critical paths", "Decouple slow operations"]
                }
            },
            {
                "id": "NFR-002",
                "category": "availability",
                "priority": "critical" if subdomain.type == SubdomainType.CORE else "high",
                "constraint": f"{baseline['availability_percent']}% availability",
                "acceptance_criteria": [
                    f"Uptime SLA: {baseline['availability_percent']}%",
                    "Graceful degradation on dependency failure"
                ],
                "implications": {
                    "for_fr": ["Include fallback strategies", "Design for resilience"],
                    "for_api": ["Implement circuit breakers", "Health checks required"],
                    "for_events": ["Use DLQ for failed messages", "Retry logic required"]
                }
            },
            {
                "id": "NFR-003",
                "category": "scalability",
                "priority": "high",
                "constraint": f"Handle {baseline['requests_per_second']} requests/second",
                "acceptance_criteria": [
                    f"Support {baseline['requests_per_second']} req/s at P95 latency",
                    "Horizontal scaling capability"
                ],
                "implications": {
                    "for_fr": ["Stateless functional requirements preferred", "Cacheable operations"],
                    "for_api": ["Rate limiting required", "Pagination for list endpoints"],
                    "for_events": ["Fan-out pattern for parallel processing", "Event consumers must be scalable"]
                }
            },
            {
                "id": "NFR-004",
                "category": "security",
                "priority": "critical" if subdomain.type == SubdomainType.CORE else "high",
                "constraint": "Industry-standard security controls",
                "acceptance_criteria": [
                    "Authentication required for all endpoints",
                    "Input validation on all requests",
                    "Audit logging for sensitive operations"
                ],
                "implications": {
                    "for_fr": ["Include security validation in requirements", "Data encryption requirements"],
                    "for_api": ["JWT/OAuth authentication", "HTTPS only"],
                    "for_events": ["Encrypt sensitive event payloads", "Validate event sources"]
                }
            }
        ]


# ============================================================================
# PHASE 1: FUNCTIONAL REQUIREMENTS GENERATION (CONSTRAINT-GUIDED)
# ============================================================================

class GenerateFunctionalRequirements(dspy.Signature):
    """Phase 1: Generate functional requirements guided by NFR constraints"""
    
    service_name: str = dspy.InputField(desc="Service name")
    service_description: str = dspy.InputField(desc="Detailed service description")
    responsibilities: str = dspy.InputField(desc="Main responsibilities (list)")
    bounded_context: str = dspy.InputField(desc="DDD Bounded Context")
    nfr_constraints: str = dspy.InputField(
        desc="NFR constraints from Phase 0 (JSON). CRITICAL: Use these to guide FR design."
    )
    
    requirements_json: str = dspy.OutputField(
        desc="""JSON array of functional requirements that RESPECT NFR constraints. EXACT format:
[
  {
    "id": "FR-001",
    "type": "functional",
    "priority": "critical|high|medium|low",
    "title": "Concise title (max 100 chars)",
    "description": "How this requirement respects NFR constraints (min 50 chars)",
    "acceptance_criteria": ["criterion 1", "criterion 2", "criterion 3"],
    "related_requirements": [],
    "nfr_dependencies": ["NFR-001", "NFR-002"],
    "implementation_hint": "Atomic operation, <Xms expected, sync/async strategy"
  }
]

CRITICAL RULES:
1. Each FR must be ATOMIC and fit within NFR latency constraints
2. Reference specific NFR IDs in nfr_dependencies
3. For CORE services: break complex operations into smaller FRs
4. Include implementation hints (sync vs async, expected latency)

Example for payment service with P95 < 200ms:
- FR-001: "Validate payment input" (50ms, sync) [NFR-001]
- FR-002: "Initiate payment async" (async) [NFR-001, NFR-003]
- FR-003: "Persist transaction" (100ms, sync) [NFR-001, NFR-002]

Generate 3-5 requirements. ONLY valid JSON."""
    )


class FunctionalRequirementsGenerator(dspy.Module):
    """Phase 1: Generate constraint-guided functional requirements"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateFunctionalRequirements)
    
    def forward(
        self,
        subdomain: Subdomain,
        nfr_constraints: List[Dict]
    ) -> List[Dict]:
        """
        Generate functional requirements that respect NFR constraints.
        
        Args:
            subdomain: Subdomain definition
            nfr_constraints: NFR constraints from Phase 0
            
        Returns:
            List of functional requirement dicts
        """
        try:
            response = self.generate(
                service_name=subdomain.name,
                service_description=subdomain.description,
                responsibilities="\n".join(f"- {r}" for r in subdomain.responsibilities),
                bounded_context=subdomain.bounded_context,
                nfr_constraints=serialize_context(nfr_constraints)
            )
            
            json_str = extract_json_from_text(response.requirements_json)
            requirements, _ = json.JSONDecoder().raw_decode(json_str)
            
            if not isinstance(requirements, list):
                requirements = [requirements]
            
            # Validate structure
            for req in requirements:
                if "id" not in req or "type" not in req:
                    raise ValueError("Invalid requirement structure")
            
            return requirements
            
        except Exception as e:
            print(f"⚠️  Phase 1 (Functional Requirements) error: {e}")
            # Fallback: Generate FR that respects NFR constraints
            return self._generate_fallback_fr(subdomain, nfr_constraints)
    
    def _generate_fallback_fr(self, subdomain: Subdomain, nfr_constraints: List[Dict]) -> List[Dict]:
        """Generate fallback FR that respects NFR constraints"""
        perf_nfr = next((nfr for nfr in nfr_constraints if nfr["category"] == "performance"), None)
        
        return [
            {
                "id": "FR-001",
                "type": "functional",
                "priority": "high",
                "title": f"Implement {subdomain.name} core functionality",
                "description": f"Implement core business logic for {subdomain.description[:100]}. Designed to respect NFR constraints.",
                "acceptance_criteria": [
                    "Service implements main responsibilities",
                    "Business rules are enforced",
                    "Data validation is in place",
                    f"Operations complete within {perf_nfr['constraint'] if perf_nfr else 'target'} latency"
                ],
                "related_requirements": [],
                "nfr_dependencies": [nfr["id"] for nfr in nfr_constraints[:2]],
                "implementation_hint": "Atomic operation, sync processing for core path"
            }
        ]


# ============================================================================
# PHASE 2: API ENDPOINTS GENERATION (FR + NFR AWARE)
# ============================================================================

class GenerateAPIEndpoints(dspy.Signature):
    """Phase 2: Generate API endpoints that implement FR within NFR constraints"""
    
    service_name: str = dspy.InputField(desc="Service name")
    functional_requirements: str = dspy.InputField(desc="Functional requirements from Phase 1 (JSON)")
    nfr_constraints: str = dspy.InputField(desc="NFR constraints from Phase 0 (JSON)")
    
    endpoints_json: str = dspy.OutputField(
        desc="""JSON array of REST API endpoints aligned with FR and NFR. EXACT format:
[
  {
    "method": "GET|POST|PUT|DELETE|PATCH",
    "path": "/api/v1/resource",
    "description": "Endpoint description",
    "request_schema": {"field": "type"},
    "response_schema": {"field": "type"},
    "authentication_required": true,
    "rate_limit": "100/min",
    "timeout_ms": 150,
    "implementation_strategy": "sync|async|eventual_consistency",
    "fr_coverage": ["FR-001", "FR-002"],
    "nfr_alignment": "Specific explanation of how this respects NFR constraints"
  }
]

CRITICAL RULES:
1. timeout_ms must align with NFR latency constraints (typically 75% of P95 target)
2. Use implementation_strategy:
   - "sync": immediate response, must fit in latency budget
   - "async": return 202, process in background via events
   - "eventual_consistency": return immediately, eventual result
3. Each endpoint must reference FR it implements
4. Include nfr_alignment explanation

Example for service with P95 < 200ms:
- GET endpoint: timeout 150ms, sync (simple read)
- POST endpoint: timeout 150ms, async (returns 202, processes via events)

Generate 3-5 REST endpoints. ONLY valid JSON."""
    )


class APIEndpointsGenerator(dspy.Module):
    """Phase 2: Generate NFR-aware API endpoints"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAPIEndpoints)
    
    def forward(
        self,
        subdomain: Subdomain,
        functional_requirements: List[Dict],
        nfr_constraints: List[Dict]
    ) -> List[Dict]:
        """
        Generate API endpoints that implement FR within NFR constraints.
        
        Args:
            subdomain: Subdomain definition
            functional_requirements: Output from Phase 1
            nfr_constraints: NFR constraints from Phase 0
            
        Returns:
            List of API endpoint dicts
        """
        try:
            response = self.generate(
                service_name=subdomain.name,
                functional_requirements=serialize_context(functional_requirements),
                nfr_constraints=serialize_context(nfr_constraints)
            )
            
            json_str = extract_json_from_text(response.endpoints_json)
            endpoints, _ = json.JSONDecoder().raw_decode(json_str)
            
            if not isinstance(endpoints, list):
                endpoints = [endpoints]
            
            # Validate structure
            for endpoint in endpoints:
                if "method" not in endpoint or "path" not in endpoint:
                    raise ValueError("Invalid endpoint structure")
            
            return endpoints
            
        except Exception as e:
            print(f"⚠️  Phase 2 (API Endpoints) error: {e}")
            # Fallback: Generate endpoints aligned with NFR
            return self._generate_fallback_endpoints(subdomain, functional_requirements, nfr_constraints)
    
    def _generate_fallback_endpoints(
        self,
        subdomain: Subdomain,
        functional_requirements: List[Dict],
        nfr_constraints: List[Dict]
    ) -> List[Dict]:
        """Generate fallback endpoints that respect NFR constraints"""
        perf_nfr = next((nfr for nfr in nfr_constraints if nfr["category"] == "performance"), None)
        timeout_ms = 150  # Default
        
        if perf_nfr and "latency" in perf_nfr["constraint"].lower():
            # Extract latency value and set timeout to 75%
            import re
            match = re.search(r'(\d+)\s*ms', perf_nfr["constraint"])
            if match:
                timeout_ms = int(int(match.group(1)) * 0.75)
        
        fr_ids = [fr["id"] for fr in functional_requirements[:2]]
        
        return [
            {
                "method": "POST",
                "path": f"/api/v1/{subdomain.name}",
                "description": f"Create {subdomain.name}",
                "request_schema": {"data": "object"},
                "response_schema": {"id": "string", "status": "string"},
                "authentication_required": True,
                "rate_limit": "100/min",
                "timeout_ms": timeout_ms,
                "implementation_strategy": "async",
                "fr_coverage": fr_ids,
                "nfr_alignment": f"Returns 202 immediately to respect {timeout_ms}ms timeout, processes async"
            },
            {
                "method": "GET",
                "path": f"/api/v1/{subdomain.name}/{{id}}",
                "description": f"Get {subdomain.name} by ID",
                "request_schema": None,
                "response_schema": {"id": "string", "data": "object"},
                "authentication_required": True,
                "rate_limit": "100/min",
                "timeout_ms": timeout_ms,
                "implementation_strategy": "sync",
                "fr_coverage": fr_ids,
                "nfr_alignment": f"Simple read operation within {timeout_ms}ms budget"
            }
        ]


# ============================================================================
# PHASE 3: DOMAIN EVENTS GENERATION (SCALABILITY + DECOUPLING)
# ============================================================================

class GenerateDomainEvents(dspy.Signature):
    """Phase 3: Generate domain events that enable scalability and decoupling"""
    
    service_name: str = dspy.InputField(desc="Service name")
    functional_requirements: str = dspy.InputField(desc="Functional requirements from Phase 1 (JSON)")
    api_endpoints: str = dspy.InputField(desc="API endpoints from Phase 2 (JSON)")
    nfr_constraints: str = dspy.InputField(
        desc="NFR constraints from Phase 0 (JSON). Focus on scalability category."
    )
    
    events_json: str = dspy.OutputField(
        desc="""JSON array of domain events that support scalability NFR. EXACT format:
[
  {
    "event_name": "EntityCreatedEvent",
    "event_type": "domain|integration|notification",
    "payload_schema": {"entity_id": "string", "timestamp": "string", "data": "object"},
    "trigger_conditions": ["Condition 1", "Condition 2"],
    "consumers": ["service-name-1", "service-name-2"],
    "purpose": "Why this event enables scalability (e.g., fan-out, parallel processing)",
    "reliability_strategy": "at_least_once|exactly_once|best_effort",
    "fr_linked": ["FR-001", "FR-002"],
    "api_linked": ["POST /api/v1/resource"],
    "nfr_support": "How this event helps achieve NFR (e.g., async processing for scalability)"
  }
]

CRITICAL RULES:
1. Events should enable SCALABILITY (fan-out, parallel processing)
2. Events DECOUPLE services (avoid cascading failures)
3. Define reliability_strategy with DLQ for critical events
4. Link events to FR and API endpoints
5. Explain nfr_support (how event achieves NFR goals)

Example for payment service:
- PaymentProcessedEvent: enables parallel processing by order, notification, analytics services
- reliability: at_least_once with DLQ
- purpose: fan-out work to multiple consumers for scalability

Generate 2-4 events. ONLY valid JSON."""
    )


class DomainEventsGenerator(dspy.Module):
    """Phase 3: Generate events for scalability and decoupling"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateDomainEvents)
    
    def forward(
        self,
        subdomain: Subdomain,
        functional_requirements: List[Dict],
        api_endpoints: List[Dict],
        nfr_constraints: List[Dict]
    ) -> List[Dict]:
        """
        Generate domain events that support scalability NFR.
        
        Args:
            subdomain: Subdomain definition
            functional_requirements: Output from Phase 1
            api_endpoints: Output from Phase 2
            nfr_constraints: NFR constraints from Phase 0
            
        Returns:
            List of domain event dicts
        """
        try:
            response = self.generate(
                service_name=subdomain.name,
                functional_requirements=serialize_context(functional_requirements),
                api_endpoints=serialize_context(api_endpoints),
                nfr_constraints=serialize_context(nfr_constraints)
            )
            
            json_str = extract_json_from_text(response.events_json)
            events, _ = json.JSONDecoder().raw_decode(json_str)
            
            if not isinstance(events, list):
                events = [events]
            
            # Validate structure
            for event in events:
                if "event_name" not in event:
                    raise ValueError("Invalid event structure")
            
            return events
            
        except Exception as e:
            print(f"⚠️  Phase 3 (Domain Events) error: {e}")
            # Fallback: Generate events that support scalability
            return self._generate_fallback_events(subdomain, functional_requirements, api_endpoints, nfr_constraints)
    
    def _generate_fallback_events(
        self,
        subdomain: Subdomain,
        functional_requirements: List[Dict],
        api_endpoints: List[Dict],
        nfr_constraints: List[Dict]
    ) -> List[Dict]:
        """Generate fallback events that support scalability"""
        event_name = f"{subdomain.name.title().replace('-', '').replace('_', '')}Event"
        fr_ids = [fr["id"] for fr in functional_requirements[:2]]
        api_paths = [f"{ep['method']} {ep['path']}" for ep in api_endpoints[:2]]
        
        return [
            {
                "event_name": event_name,
                "event_type": "domain",
                "payload_schema": {
                    "entity_id": "string",
                    "timestamp": "string",
                    "data": "object",
                    "metadata": "object"
                },
                "trigger_conditions": [
                    f"When {subdomain.name} entity is created or updated",
                    "After successful validation and persistence"
                ],
                "consumers": subdomain.dependencies or ["downstream-services"],
                "purpose": "Enable parallel processing by multiple consumers for scalability",
                "reliability_strategy": "at_least_once",
                "fr_linked": fr_ids,
                "api_linked": api_paths,
                "nfr_support": "Async event processing enables horizontal scaling and decouples services"
            }
        ]


# ============================================================================
# PHASE 4: CORRELATION VALIDATION
# ============================================================================

class CorrelationValidator(dspy.Module):
    """Phase 4: Validate correlation and coherence across all phases"""
    
    def forward(
        self,
        subdomain: Subdomain,
        nfr_constraints: List[Dict],
        functional_requirements: List[Dict],
        api_endpoints: List[Dict],
        domain_events: List[Dict]
    ) -> Dict:
        """
        Validate correlation between all phases.
        
        Args:
            subdomain: Subdomain definition
            nfr_constraints: Phase 0 output
            functional_requirements: Phase 1 output
            api_endpoints: Phase 2 output
            domain_events: Phase 3 output
            
        Returns:
            Validation report with errors, warnings, and metrics
        """
        errors = []
        warnings = []
        
        # 1. NFR → FR Correlation
        nfr_ids = {nfr["id"] for nfr in nfr_constraints}
        fr_nfr_refs = set()
        for fr in functional_requirements:
            fr_nfr_refs.update(fr.get("nfr_dependencies", []))
        
        nfr_to_fr_coverage = (len(fr_nfr_refs) / len(nfr_ids) * 100) if nfr_ids else 0
        
        if nfr_to_fr_coverage < 50:
            warnings.append(f"Low NFR→FR coverage: {nfr_to_fr_coverage:.1f}%. Not all NFR constraints referenced by FRs")
        
        # Check if FRs respect latency constraints
        perf_nfr = next((nfr for nfr in nfr_constraints if nfr["category"] == "performance"), None)
        if perf_nfr:
            for fr in functional_requirements:
                if "implementation_hint" not in fr:
                    warnings.append(f"{fr['id']}: Missing implementation_hint to validate latency compliance")
        
        # 2. FR → API Correlation
        fr_ids = {fr["id"] for fr in functional_requirements}
        api_fr_refs = set()
        for api in api_endpoints:
            api_fr_refs.update(api.get("fr_coverage", []))
        
        fr_to_api_coverage = (len(api_fr_refs) / len(fr_ids) * 100) if fr_ids else 0
        
        if fr_to_api_coverage < 80:
            warnings.append(f"Low FR→API coverage: {fr_to_api_coverage:.1f}%. Some FRs not covered by APIs")
        
        # Check if API timeouts align with NFR
        if perf_nfr:
            for api in api_endpoints:
                if "timeout_ms" not in api:
                    warnings.append(f"{api['method']} {api['path']}: Missing timeout_ms")
        
        # 3. API → Events Correlation
        async_apis = [api for api in api_endpoints if api.get("implementation_strategy") == "async"]
        
        if len(async_apis) > 0 and len(domain_events) == 0:
            errors.append("Async APIs defined but no domain events generated")
        
        api_to_event_coverage = (len(domain_events) / len(async_apis) * 100) if async_apis else 100
        
        # 4. Overall Consistency
        tech_stack_valid = True  # Simplified validation
        
        # Calculate overall consistency score
        overall_score = (nfr_to_fr_coverage + fr_to_api_coverage + api_to_event_coverage) / 3
        
        # Generate recommendations
        recommendations = []
        if nfr_to_fr_coverage < 80:
            recommendations.append("Ensure all FRs reference relevant NFR constraints")
        if fr_to_api_coverage < 80:
            recommendations.append("Add API endpoints to cover all functional requirements")
        if len(async_apis) > len(domain_events):
            recommendations.append("Add domain events for async API endpoints")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "correlation_metrics": {
                "nfr_to_fr_coverage": round(nfr_to_fr_coverage, 1),
                "fr_to_api_coverage": round(fr_to_api_coverage, 1),
                "api_to_event_coverage": round(api_to_event_coverage, 1),
                "overall_consistency_score": round(overall_score, 1)
            },
            "recommendations": recommendations
        }


# ============================================================================
# PIPELINE - 5-PHASE SEQUENTIAL EXECUTION
# ============================================================================

class SpecificationGeneratorPipeline(dspy.Module):
    """
    5-Phase NFR-First Pipeline
    
    Flow:
    Phase 0: NFR Analysis → Constraints
    Phase 1: FR Generation (with NFR constraints) → Requirements
    Phase 2: API Generation (with FR + NFR) → Endpoints
    Phase 3: Events Generation (with FR + API + NFR) → Events
    Phase 4: Validation (all phases) → Validation Report
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize LLM engine
        print("🔧 Initializing LLM engine...")
        try:
            initialize_llm_engine()
            print("✅ LLM engine initialized successfully")
        except Exception as e:
            print(f"⚠️  LLM initialization warning: {e}")
            print("   Pipeline will attempt to use default LLM configuration")
        
        # Initialize all 5 phases
        self.phase0_nfr = NonFunctionalRequirementsAnalyzer()
        self.phase1_fr = FunctionalRequirementsGenerator()
        self.phase2_api = APIEndpointsGenerator()
        self.phase3_events = DomainEventsGenerator()
        self.phase4_validation = CorrelationValidator()
    
    def forward(
        self,
        subdomain: Subdomain,
        technical_stack: Dict[str, str] = None,
        global_constraints: Dict[str, str] = None
    ) -> Dict:
        """
        Execute the 5-phase pipeline with NFR-first approach.
        
        Args:
            subdomain: Subdomain to generate specs for
            technical_stack: Technical stack preferences
            global_constraints: Global project constraints
            
        Returns:
            Complete microservice specification with validation report
        """
        service_name = subdomain.name
        tech_stack = technical_stack or DEFAULT_TECH_STACK
        constraints = global_constraints or {}
        
        print(f"\n{'='*60}")
        print(f"🎯 Generating specs for: {service_name}")
        print(f"   Service Type: {subdomain.type.value}")
        print(f"{'='*60}")
        
        # PHASE 0: NFR Constraints Analysis (PRIMARY)
        print(f"🔍 [0/5] Phase 0: Analyzing NFR constraints...")
        nfr_constraints = self.phase0_nfr(subdomain, constraints, tech_stack)
        print(f"   ✅ Extracted {len(nfr_constraints)} NFR constraints")
        
        # PHASE 1: Functional Requirements (Constraint-Guided)
        print(f"📋 [1/5] Phase 1: Generating functional requirements (NFR-guided)...")
        functional_reqs = self.phase1_fr(subdomain, nfr_constraints)
        print(f"   ✅ Generated {len(functional_reqs)} functional requirements")
        
        # PHASE 2: API Endpoints (FR + NFR Aware)
        print(f"🔌 [2/5] Phase 2: Generating API endpoints (FR + NFR aware)...")
        api_endpoints = self.phase2_api(subdomain, functional_reqs, nfr_constraints)
        print(f"   ✅ Generated {len(api_endpoints)} API endpoints")
        
        # PHASE 3: Domain Events (Scalability + Decoupling)
        print(f"📡 [3/5] Phase 3: Generating domain events (scalability focus)...")
        events_published = self.phase3_events(subdomain, functional_reqs, api_endpoints, nfr_constraints)
        print(f"   ✅ Generated {len(events_published)} domain events")
        
        # PHASE 4: Correlation Validation
        print(f"✓ [4/5] Phase 4: Validating correlation and coherence...")
        validation_report = self.phase4_validation(
            subdomain, nfr_constraints, functional_reqs, api_endpoints, events_published
        )
        print(f"   ✅ Validation complete: {validation_report['valid']}")
        if validation_report['warnings']:
            print(f"   ⚠️  {len(validation_report['warnings'])} warnings")
        if validation_report['errors']:
            print(f"   ❌ {len(validation_report['errors'])} errors")
        print(f"   📊 Overall Score: {validation_report['correlation_metrics']['overall_consistency_score']:.1f}%")
        
        print(f"{'='*60}\n")
        
        # Convert NFR constraints to standard Requirement format for output
        nfr_requirements = self._convert_nfr_constraints_to_requirements(nfr_constraints)
        
        return {
            "service_name": service_name,
            "version": "1.0.0",
            "description": subdomain.description,
            "bounded_context": subdomain.bounded_context,
            "functional_requirements": functional_reqs,
            "non_functional_requirements": nfr_requirements,
            "events_published": events_published,
            "events_subscribed": [],
            "api_endpoints": api_endpoints,
            "message_queues": [],
            "dependencies": self._convert_dependencies(subdomain),
            "technology_stack": tech_stack,
            "infrastructure_requirements": INFRASTRUCTURE_TEMPLATE,
            "monitoring_requirements": MONITORING_REQUIREMENTS,
            "generated_at": datetime.utcnow(),
            "generated_by": "agent3-spec-generator-5phase-nfr-first",
            "validation_report": validation_report
        }
    
    def _convert_nfr_constraints_to_requirements(self, nfr_constraints: List[Dict]) -> List[Dict]:
        """Convert NFR constraints to standard Requirement format"""
        requirements = []
        for nfr in nfr_constraints:
            requirements.append({
                "id": nfr["id"],
                "type": "non_functional",
                "priority": nfr["priority"],
                "title": f"{nfr['category'].title()}: {nfr['constraint']}",
                "description": f"Category: {nfr['category']}. Constraint: {nfr['constraint']}",
                "acceptance_criteria": nfr["acceptance_criteria"],
                "related_requirements": []
            })
        return requirements
    
    def _convert_dependencies(self, subdomain: Subdomain) -> List[Dict]:
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
# ORCHESTRATOR - Streaming with Phase-Level Progress
# ============================================================================

class SpecificationOrchestrator:
    """Orchestrator with SSE streaming and phase-level progress reporting"""
    
    def __init__(self):
        """Initialize orchestrator and pipeline with LLM"""
        print("🚀 Initializing SpecificationOrchestrator (5-Phase NFR-First)...")
        self.pipeline = SpecificationGeneratorPipeline()
        print("✅ Orchestrator ready")
    
    async def generate_all_specs_streaming(
        self, 
        architecture_input: ArchitectureInput
    ) -> AsyncGenerator[dict, None]:
        """
        Generate specifications with SSE streaming and phase-level progress.
        Yields SSE events for each phase with detailed progress.
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
                    "message": f"Starting 5-phase generation for {subdomain_name}"
                })
            }
            
            try:
                # Phase 0: NFR Analysis
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "phase": 0,
                        "phase_name": "NFR Analysis",
                        "step_number": "0/5",
                        "message": "Phase 0: Extracting non-functional constraints..."
                    })
                }
                
                # Phase 1: Functional Requirements
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "phase": 1,
                        "phase_name": "Functional Requirements",
                        "step_number": "1/5",
                        "message": "Phase 1: Generating constraint-guided functional requirements..."
                    })
                }
                
                # Phase 2: API Endpoints
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "phase": 2,
                        "phase_name": "API Endpoints",
                        "step_number": "2/5",
                        "message": "Phase 2: Designing NFR-aware API endpoints..."
                    })
                }
                
                # Phase 3: Domain Events
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "phase": 3,
                        "phase_name": "Domain Events",
                        "step_number": "3/5",
                        "message": "Phase 3: Generating scalability-focused domain events..."
                    })
                }
                
                # Phase 4: Validation
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "phase": 4,
                        "phase_name": "Correlation Validation",
                        "step_number": "4/5",
                        "message": "Phase 4: Validating correlation and coherence..."
                    })
                }
                
                # Execute pipeline with 5-phase architecture
                spec = self.pipeline(
                    subdomain=subdomain,
                    technical_stack=architecture_input.technical_stack or {},
                    global_constraints=architecture_input.global_constraints or {}
                )
                
                microservices.append(spec)
                
                # Event: phase complete (with validation metrics)
                yield {
                    "event": "phase_complete",
                    "data": json.dumps({
                        "subdomain": subdomain_name,
                        "phase": 4,
                        "validation": "✓ PASS" if spec["validation_report"]["valid"] else "⚠ WARNINGS",
                        "metrics": spec["validation_report"]["correlation_metrics"]
                    })
                }
                
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
                        "message": f"✅ {subdomain_name} completed with 5-phase NFR-first architecture",
                        "progress_percent": int((idx / total) * 100),
                        "validation_score": spec["validation_report"]["correlation_metrics"]["overall_consistency_score"]
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
            "specification_version": "2.0.0",
            "architecture": "5-phase-nfr-first",
            "microservices": microservices,
            "inter_service_communication": self._build_communication_map(architecture_input),
            "shared_infrastructure": SHARED_INFRASTRUCTURE,
            "generated_at": datetime.utcnow(),
            "agent_version": "2.0.0"
        }
        
        yield {
            "event": "complete",
            "data": json.dumps(final_output, default=str)
        }
    
    def _build_communication_map(self, architecture_input: ArchitectureInput) -> Dict[str, List[str]]:
        """Build inter-service communication map"""
        comm_map = {}
        for subdomain in architecture_input.subdomains:
            comm_map[subdomain.name] = subdomain.dependencies or []
        return comm_map
    
    def _generate_minimal_spec(self, subdomain: Subdomain) -> Dict:
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
            "generated_by": "agent3-fallback",
            "validation_report": {
                "valid": False,
                "errors": ["Fallback spec used due to generation failure"],
                "warnings": [],
                "correlation_metrics": {
                    "nfr_to_fr_coverage": 0,
                    "fr_to_api_coverage": 0,
                    "api_to_event_coverage": 0,
                    "overall_consistency_score": 0
                },
                "recommendations": ["Retry generation with valid inputs"]
            }
        }