"""
Core Business Logic - DSPy Pipeline for Specification Generation
"""

import json
from datetime import datetime
from typing import Any

import dspy

from app.models.input_schema import ArchitectureInput, Subdomain
from app.models.output_schema import (
    RequirementType,
)
from app.signatures.spec_signature import GenerateMicroserviceSpec, ValidateSpecificationQuality


class SpecificationGeneratorPipeline(dspy.Module):
    """
    DSPy Module for generating functional specifications.
    Uses Chain-of-Thought reasoning with validation.
    """

    def __init__(self):
        super().__init__()
        # Chain of Thought for structured reasoning
        self.generate_spec = dspy.ChainOfThought(GenerateMicroserviceSpec)
        self.validate_spec = dspy.ChainOfThought(ValidateSpecificationQuality)

    def forward(
        self,
        subdomain: Subdomain,
        technical_stack: dict[str, str],
        global_constraints: dict[str, str],
    ) -> dict[str, Any]:
        """
        Generate specification for a single subdomain.

        Args:
            subdomain: Subdomain definition
            technical_stack: Technology preferences
            global_constraints: Project-wide constraints

        Returns:
            Structured specification dictionary
        """
        # Prepare inputs
        responsibilities_str = ", ".join(subdomain.responsibilities)
        dependencies_str = ", ".join(subdomain.dependencies) if subdomain.dependencies else "None"
        comm_patterns = ", ".join([p.value for p in subdomain.communication_patterns])
        tech_constraints = json.dumps(technical_stack) + " | " + json.dumps(global_constraints)

        # Generate specification
        result = self.generate_spec(
            subdomain_name=subdomain.name,
            subdomain_description=subdomain.description,
            responsibilities=responsibilities_str,
            dependencies=dependencies_str,
            communication_patterns=comm_patterns,
            technical_constraints=tech_constraints,
        )

        # Parse and structure output
        spec_data = {
            "service_name": subdomain.name,
            "version": "1.0.0",
            "description": subdomain.description,
            "bounded_context": subdomain.bounded_context,
            "functional_requirements": self._parse_requirements(
                result.functional_requirements, RequirementType.FUNCTIONAL
            ),
            "non_functional_requirements": self._parse_requirements(
                result.non_functional_requirements, RequirementType.NON_FUNCTIONAL
            ),
            "events_published": self._parse_events(result.events_published),
            "events_subscribed": self._parse_events(result.events_subscribed),
            "api_endpoints": self._parse_endpoints(result.api_endpoints),
            "dependencies": self._parse_dependencies(result.service_dependencies),
            "technology_stack": technical_stack,
            "infrastructure_requirements": self._parse_json_safe(
                result.infrastructure_requirements
            ),
            "monitoring_requirements": [
                "Prometheus metrics exposition",
                "Structured logging (JSON)",
                "Distributed tracing (OpenTelemetry)",
                "Health check endpoint",
            ],
            "generated_at": datetime.utcnow(),
            "generated_by": "agent3-spec-generator",
        }

        # Validate quality
        validation_checklist = """
        - All functional requirements have acceptance criteria
        - Events have complete payload schemas
        - API endpoints have request/response schemas
        - Dependencies specify fallback strategies
        - Non-functional requirements include performance metrics
        """

        validation = self.validate_spec(
            specification=json.dumps(spec_data, default=str),
            requirements_checklist=validation_checklist,
        )

        spec_data["validation_score"] = self._extract_score(validation.validation_result)
        spec_data["validation_notes"] = validation.recommendations

        return spec_data

    def _parse_requirements(self, raw_text: str, req_type: RequirementType) -> list[dict]:
        """Parse requirements from LLM output"""
        try:
            reqs = json.loads(raw_text) if raw_text.strip().startswith("[") else []
            if not isinstance(reqs, list):
                reqs = [reqs]

            parsed = []
            for idx, req in enumerate(reqs):
                if isinstance(req, dict):
                    parsed.append(
                        {
                            "id": req.get("id", f"{req_type.value.upper()[:2]}-{idx+1:03d}"),
                            "type": req_type.value,
                            "priority": req.get("priority", "medium"),
                            "title": req.get("title", f"Requirement {idx+1}"),
                            "description": req.get("description", ""),
                            "acceptance_criteria": req.get("acceptance_criteria", []),
                            "related_requirements": req.get("related_requirements", []),
                        }
                    )
            return parsed
        except:
            # Fallback: Generate minimal requirement
            return [
                {
                    "id": f"{req_type.value.upper()[:2]}-001",
                    "type": req_type.value,
                    "priority": "high",
                    "title": f"Generated {req_type.value} requirement",
                    "description": raw_text[:200],
                    "acceptance_criteria": ["Requirement must be implemented as specified"],
                    "related_requirements": [],
                }
            ]

    def _parse_events(self, raw_text: str) -> list[dict]:
        """Parse event definitions"""
        try:
            events = json.loads(raw_text) if raw_text.strip().startswith("[") else []
            if not isinstance(events, list):
                events = [events]
            return [e for e in events if isinstance(e, dict) and "event_name" in e]
        except:
            return []

    def _parse_endpoints(self, raw_text: str) -> list[dict]:
        """Parse API endpoints"""
        try:
            endpoints = json.loads(raw_text) if raw_text.strip().startswith("[") else []
            if not isinstance(endpoints, list):
                endpoints = [endpoints]
            return [e for e in endpoints if isinstance(e, dict) and "method" in e and "path" in e]
        except:
            return []

    def _parse_dependencies(self, raw_text: str) -> list[dict]:
        """Parse service dependencies"""
        try:
            deps = json.loads(raw_text) if raw_text.strip().startswith("[") else []
            if not isinstance(deps, list):
                deps = [deps]
            return [d for d in deps if isinstance(d, dict) and "service_name" in d]
        except:
            return []

    def _parse_json_safe(self, raw_text: str) -> dict[str, Any]:
        """Safe JSON parsing with fallback"""
        try:
            return json.loads(raw_text) if raw_text.strip().startswith("{") else {}
        except:
            return {"raw_content": raw_text}

    def _extract_score(self, validation_text: str) -> int:
        """Extract validation score from text"""
        try:
            import re

            match = re.search(r"\b(\d{1,3})\b", validation_text)
            return int(match.group(1)) if match else 85
        except:
            return 85


class SpecificationOrchestrator:
    """
    High-level orchestrator for generating all specifications.
    """

    def __init__(self):
        self.pipeline = SpecificationGeneratorPipeline()

    def generate_all_specs(self, architecture_input: ArchitectureInput) -> dict[str, Any]:
        """
        Generate specifications for all subdomains.

        Returns:
            Complete functional specification output
        """
        microservices = []
        inter_service_comm = {}

        for subdomain in architecture_input.subdomains:
            spec = self.pipeline(
                subdomain=subdomain,
                technical_stack=architecture_input.technical_stack or {},
                global_constraints=architecture_input.global_constraints or {},
            )
            microservices.append(spec)

            # Build communication map
            service_name = subdomain.name
            inter_service_comm[service_name] = subdomain.dependencies or []

        return {
            "project_name": architecture_input.project_name,
            "project_description": architecture_input.project_description,
            "specification_version": "1.0.0",
            "microservices": microservices,
            "inter_service_communication": inter_service_comm,
            "shared_infrastructure": {
                "message_broker": "RabbitMQ / Kafka",
                "api_gateway": "Kong / Nginx",
                "service_mesh": "Istio (optional)",
                "monitoring": "Prometheus + Grafana",
                "logging": "ELK Stack",
                "tracing": "Jaeger",
            },
            "generated_at": datetime.utcnow(),
            "agent_version": "1.0.0",
        }
