"""
DSPy Signatures - Declarative Prompt Definitions
"""

import dspy


class GenerateMicroserviceSpec(dspy.Signature):
    """
    Generate detailed functional specification for a microservice.

    This signature defines the contract between input (subdomain architecture)
    and output (detailed functional specification).
    """

    # Input Fields
    subdomain_name: str = dspy.InputField(desc="Name of the subdomain/microservice")
    subdomain_description: str = dspy.InputField(desc="High-level description of the subdomain")
    responsibilities: str = dspy.InputField(desc="List of responsibilities (comma-separated)")
    dependencies: str = dspy.InputField(desc="Dependent services (comma-separated)")
    communication_patterns: str = dspy.InputField(
        desc="Communication patterns (REST, Events, Queue)"
    )
    technical_constraints: str = dspy.InputField(desc="Technical stack and constraints")

    # Output Fields
    functional_requirements: str = dspy.OutputField(
        desc="Detailed functional requirements in structured JSON format with id, type, priority, title, description, acceptance_criteria"
    )
    non_functional_requirements: str = dspy.OutputField(
        desc="Non-functional requirements (performance, security, scalability) in JSON format"
    )
    events_published: str = dspy.OutputField(
        desc="Events published by this service in JSON format with event_name, event_type, payload_schema, trigger_conditions"
    )
    events_subscribed: str = dspy.OutputField(
        desc="Events subscribed from other services in JSON format"
    )
    api_endpoints: str = dspy.OutputField(
        desc="REST API endpoints in JSON format with method, path, description, request_schema, response_schema"
    )
    service_dependencies: str = dspy.OutputField(
        desc="Service dependencies in JSON format with service_name, dependency_type, communication_method, criticality"
    )
    infrastructure_requirements: str = dspy.OutputField(
        desc="Infrastructure needs (database, cache, queue) in JSON format"
    )


class ValidateSpecificationQuality(dspy.Signature):
    """
    Validate the quality and completeness of generated specifications.
    """

    specification: str = dspy.InputField(desc="Generated specification in JSON format")
    requirements_checklist: str = dspy.InputField(desc="Checklist of mandatory requirements")

    validation_result: str = dspy.OutputField(
        desc="Validation result with score (0-100) and issues found"
    )
    recommendations: str = dspy.OutputField(desc="Recommendations for improvement")
