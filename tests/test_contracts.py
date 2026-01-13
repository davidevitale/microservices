"""
Contract Tests - JSON Schema Validation & Type Safety
"""
import pytest
import json
from pydantic import ValidationError
from datetime import datetime

from app.models.input_schema import (
    ArchitectureInput,
    Subdomain,
    SubdomainType,
    CommunicationPattern
)
from app.models.output_schema import (
    FunctionalSpecificationOutput,
    MicroserviceSpec,
    Requirement,
    EventDefinition,
    APIEndpoint,
    ServiceDependency,
    MessageQueueConfig,
    RequirementType,
    RequirementPriority
)


class TestInputSchemaContracts:
    """Test input schema validation and contracts"""
    
    def test_valid_subdomain_minimal(self):
        """Minimal valid subdomain passes validation"""
        subdomain = Subdomain(
            name="test-service",
            type=SubdomainType.CORE,
            description="Test service description",
            bounded_context="Test",
            responsibilities=["Test responsibility"]
        )
        
        assert subdomain.name == "test-service"
        assert subdomain.type == SubdomainType.CORE
        assert len(subdomain.responsibilities) == 1
        assert subdomain.dependencies == []
    
    def test_valid_subdomain_complete(self, sample_subdomain):
        """Complete subdomain with all fields"""
        assert sample_subdomain.name == "order-management"
        assert sample_subdomain.type == SubdomainType.CORE
        assert len(sample_subdomain.responsibilities) == 4
        assert len(sample_subdomain.dependencies) == 2
        assert CommunicationPattern.ASYNC_EVENT in sample_subdomain.communication_patterns
    
    def test_subdomain_name_validation_lowercase(self):
        """Subdomain names are converted to lowercase"""
        subdomain = Subdomain(
            name="ORDER-Service",
            type=SubdomainType.CORE,
            description="This is a valid test description",
            bounded_context="Test",
            responsibilities=["Test"]
        )
        assert subdomain.name == "order-service"
    
    def test_subdomain_name_invalid_characters(self):
        """Invalid characters in name raise error"""
        with pytest.raises(ValidationError) as exc_info:
            Subdomain(
                name="order service!@#",
                type=SubdomainType.CORE,
                description="Test",
                bounded_context="Test",
                responsibilities=["Test"]
            )
        assert "must be alphanumeric" in str(exc_info.value).lower()
    
    def test_subdomain_empty_name(self):
        """Empty name raises error"""
        with pytest.raises(ValidationError):
            Subdomain(
                name="",
                type=SubdomainType.CORE,
                description="Test",
                bounded_context="Test",
                responsibilities=["Test"]
            )
    
    def test_subdomain_short_name(self):
        """Name too short raises error"""
        with pytest.raises(ValidationError):
            Subdomain(
                name="ab",
                type=SubdomainType.CORE,
                description="Test",
                bounded_context="Test",
                responsibilities=["Test"]
            )
    
    def test_subdomain_empty_responsibilities(self):
        """Empty responsibilities list raises error"""
        with pytest.raises(ValidationError):
            Subdomain(
                name="test-service",
                type=SubdomainType.CORE,
                description="Test",
                bounded_context="Test",
                responsibilities=[]
            )
    
    def test_subdomain_short_description(self):
        """Description too short raises error"""
        with pytest.raises(ValidationError):
            Subdomain(
                name="test-service",
                type=SubdomainType.CORE,
                description="Short",
                bounded_context="Test",
                responsibilities=["Test"]
            )
    
    def test_valid_architecture_minimal(self, minimal_architecture):
        """Minimal architecture passes validation"""
        assert minimal_architecture.project_name == "minimal-service"
        assert len(minimal_architecture.subdomains) == 1
        assert minimal_architecture.global_constraints == {}
        assert minimal_architecture.technical_stack == {}
    
    def test_valid_architecture_complete(self, sample_architecture):
        """Complete architecture with all fields"""
        assert sample_architecture.project_name == "ecommerce-platform"
        assert len(sample_architecture.subdomains) == 2
        assert "max_response_time" in sample_architecture.global_constraints
        assert "language" in sample_architecture.technical_stack
    
    def test_architecture_empty_subdomains(self):
        """Empty subdomains list raises error"""
        with pytest.raises(ValidationError):
            ArchitectureInput(
                project_name="test",
                project_description="Test",
                subdomains=[]
            )
    
    def test_architecture_json_serialization(self, sample_architecture):
        """Architecture can be serialized to JSON"""
        json_str = sample_architecture.model_dump_json()
        data = json.loads(json_str)
        
        assert data["project_name"] == "ecommerce-platform"
        assert len(data["subdomains"]) == 2
    
    def test_architecture_from_dict(self):
        """Architecture can be created from dict"""
        data = {
            "project_name": "test-project",
            "project_description": "Test description",
            "subdomains": [
                {
                    "name": "test-service",
                    "type": "core",
                    "description": "Test service description",
                    "bounded_context": "Test",
                    "responsibilities": ["Test"]
                }
            ]
        }
        
        arch = ArchitectureInput(**data)
        assert arch.project_name == "test-project"


class TestOutputSchemaContracts:
    """Test output schema validation and contracts"""
    
    def test_valid_requirement_functional(self, sample_requirement):
        """Valid functional requirement"""
        assert sample_requirement.id == "FR-001"
        assert sample_requirement.type == RequirementType.FUNCTIONAL
        assert sample_requirement.priority == RequirementPriority.CRITICAL
        assert len(sample_requirement.acceptance_criteria) == 4
    
    def test_valid_requirement_non_functional(self):
        """Valid non-functional requirement"""
        req = Requirement(
            id="NFR-001",
            type=RequirementType.NON_FUNCTIONAL,
            priority=RequirementPriority.HIGH,
            title="Performance SLA",
            description="System must maintain 99.9% uptime",
            acceptance_criteria=["Uptime > 99.9%", "Max downtime 8.76h/year"]
        )
        
        assert req.type == RequirementType.NON_FUNCTIONAL
        assert req.priority == RequirementPriority.HIGH
    
    def test_requirement_short_title(self):
        """Title too short raises error"""
        with pytest.raises(ValidationError):
            Requirement(
                id="FR-001",
                type=RequirementType.FUNCTIONAL,
                priority=RequirementPriority.HIGH,
                title="Test",
                description="This is a longer description",
                acceptance_criteria=["Criteria"]
            )
    
    def test_requirement_empty_acceptance_criteria(self):
        """Empty acceptance criteria raises error"""
        with pytest.raises(ValidationError):
            Requirement(
                id="FR-001",
                type=RequirementType.FUNCTIONAL,
                priority=RequirementPriority.HIGH,
                title="Test Requirement",
                description="Test description here",
                acceptance_criteria=[]
            )
    
    def test_valid_event_definition(self, sample_event):
        """Valid event definition"""
        assert sample_event.event_name == "OrderCreatedEvent"
        assert sample_event.event_type == "domain"
        assert "order_id" in sample_event.payload_schema["properties"]
        assert len(sample_event.consumers) == 3
    
    def test_event_name_pattern(self):
        """Event name must match pattern"""
        with pytest.raises(ValidationError):
            EventDefinition(
                event_name="invalid_event_name",  # Should be PascalCase with Event suffix
                event_type="domain",
                payload_schema={},
                trigger_conditions=["Test"]
            )
    
    def test_valid_api_endpoint(self, sample_api_endpoint):
        """Valid API endpoint"""
        assert sample_api_endpoint.method == "POST"
        assert sample_api_endpoint.path == "/api/v1/orders"
        assert sample_api_endpoint.authentication_required is True
    
    def test_api_endpoint_invalid_method(self):
        """Invalid HTTP method raises error"""
        with pytest.raises(ValidationError):
            APIEndpoint(
                method="INVALID",
                path="/api/test",
                description="Test",
                response_schema={}
            )
    
    def test_api_endpoint_invalid_path(self):
        """Path must start with /"""
        with pytest.raises(ValidationError):
            APIEndpoint(
                method="GET",
                path="api/test",  # Missing leading /
                description="Test",
                response_schema={}
            )
    
    def test_valid_service_dependency(self, sample_dependency):
        """Valid service dependency"""
        assert sample_dependency.service_name == "payment-service"
        assert sample_dependency.criticality == "critical"
        assert sample_dependency.fallback_strategy is not None
    
    def test_service_dependency_invalid_criticality(self):
        """Invalid criticality pattern raises error"""
        with pytest.raises(ValidationError):
            ServiceDependency(
                service_name="test",
                dependency_type="service",
                communication_method="rest",
                criticality="invalid"  # Must be critical|high|medium|low
            )
    
    def test_valid_microservice_spec_minimal(self):
        """Minimal valid microservice spec"""
        spec = MicroserviceSpec(
            service_name="test-service",
            description="Test service description here",
            bounded_context="Test",
            functional_requirements=[],
            non_functional_requirements=[],
            technology_stack={},
            infrastructure_requirements={},
            monitoring_requirements=[]
        )
        
        assert spec.service_name == "test-service"
        assert spec.version == "1.0.0"
    
    def test_valid_microservice_spec_complete(self, sample_microservice_spec):
        """Complete microservice specification"""
        assert sample_microservice_spec.service_name == "order-service"
        assert len(sample_microservice_spec.functional_requirements) == 1
        assert len(sample_microservice_spec.non_functional_requirements) == 1
        assert len(sample_microservice_spec.events_published) == 1
        assert len(sample_microservice_spec.api_endpoints) == 1
        assert len(sample_microservice_spec.dependencies) == 1
    
    def test_microservice_spec_json_serialization(self, sample_microservice_spec):
        """Microservice spec can be serialized"""
        json_str = sample_microservice_spec.model_dump_json()
        data = json.loads(json_str)
        
        assert data["service_name"] == "order-service"
        assert "functional_requirements" in data
    
    def test_valid_functional_specification_output(self, sample_microservice_spec):
        """Complete functional specification output"""
        output = FunctionalSpecificationOutput(
            project_name="test-project",
            project_description="Test project description",
            microservices=[sample_microservice_spec],
            inter_service_communication={
                "order-service": ["payment-service"]
            }
        )
        
        assert output.project_name == "test-project"
        assert len(output.microservices) == 1
        assert output.specification_version == "1.0.0"
        assert isinstance(output.generated_at, datetime)
    
    def test_output_json_schema_generation(self):
        """Output schema can generate JSON schema"""
        schema = FunctionalSpecificationOutput.model_json_schema()
        
        assert "properties" in schema
        assert "project_name" in schema["properties"]
        assert "microservices" in schema["properties"]