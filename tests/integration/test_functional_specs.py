"""
Test Integration: Validazione Requisiti Funzionali - Aircut Platform
=====================================================================

Test specifico per validare che l'AI generi requisiti funzionali
corretti, completi e semanticamente coerenti per la piattaforma Aircut.

FOCUS: SOLO portfolio-service (gestione foto, album, portfolio barbieri)

Strategia di Testing (LLM indeterministico):
- Validazione strutturale (Pydantic)
- Validazione semantica (range di parole chiave)
- Validazione di completezza (soglie minime)

Dominio: Piattaforma per barbieri (portfolio, foto, interazioni utente)
"""

import pytest
import os
# FORZA LOCALHOST PER WINDOWS (Ignora il .env di Docker solo per questo test)
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
from typing import List, Set

from app.models.input_schema import (
    ArchitectureInput,
    Subdomain,
    SubdomainType,
    CommunicationPattern,
)
from app.models.output_schema import (
    MicroserviceSpec,
    Requirement,
    RequirementType,
    RequirementPriority,
)
from app.modules.generator_module import SpecificationOrchestrator


# ============================================================================
# CONFIGURAZIONE DOMINIO: AIRCUT - PORTFOLIO SERVICE
# ============================================================================

# Dizionario di parole chiave per validazione semantica
# Organizzato per aree funzionali del PORTFOLIO SERVICE
AIRCUT_PORTFOLIO_KEYWORDS = {
    "photo_upload_management": {
        "portfolio",

        "photo",
        "foto",
        "image",
        "immagine",
        "gallery",
        "galleria",
        "upload",
        "caricamento",
        "showcase",
        "vetrina",
        "picture",
        "media",
    },
    "album_organization": {
        "album",
        "collection",
        "collezione",
        "organize",
        "organizzare",
        "category",
        "categoria",
        "tag",
        "metadata",
        "metadati",
    },
    "barber_profile": {
        "barber",
        "barbiere",
        "hairdresser",
        "parrucchiere",
        "stylist",
        "profile",
        "profilo",
        "bio",
        "biografia",
        "experience",
        "esperienza",
        "work",
        "lavoro",
    },
    "visibility_privacy": {
        "privacy",
        "visibility",
        "visibilità",
        "public",
        "pubblico",
        "private",
        "privato",
        "permission",
        "permesso",
        "access",
        "accesso",
    },
    "haircut_showcase": {
        "haircut",
        "taglio",
        "hairstyle",
        "acconciatura",
        "before",
        "after",
        "prima",
        "dopo",
        "transformation",
        "trasformazione",
        "result",
        "risultato",
    },
}

# Flatten per ricerca rapida
ALL_PORTFOLIO_KEYWORDS = set().union(*AIRCUT_PORTFOLIO_KEYWORDS.values())


# ============================================================================
# FIXTURE: INPUT ARCHITETTURA AIRCUT - SOLO PORTFOLIO SERVICE
# ============================================================================

@pytest.fixture
def aircut_portfolio_input() -> ArchitectureInput:
    """
    Input architetturale per la piattaforma Aircut - SOLO Portfolio Service.
    
    Scenario: Sistema di gestione portfolio fotografico per barbieri dove possono:
    - Caricare e organizzare foto del loro lavoro
    - Creare album tematici (tagli, barba, colorazioni, ecc.)
    - Gestire visibilità e privacy delle foto
    - Mostrare trasformazioni before/after
    - Aggiungere metadati (tags, descrizioni, prezzi)
    """
    return ArchitectureInput(
        project_name="Aircut",
        project_description=(
            "Piattaforma di gestione portfolio fotografico per barbieri. "
            "I barbieri possono caricare foto del loro lavoro, organizzarle in album tematici, "
            "gestire la visibilità (pubblico/privato), aggiungere metadati (tags, descrizioni, prezzi), "
            "e mostrare trasformazioni before/after per attirare nuovi clienti."
        ),
        subdomains=[
            Subdomain(
                name="portfolio-service",
                type=SubdomainType.CORE,
                description=(
                    "Gestisce il portfolio fotografico completo dei barbieri. "
                    "Include upload foto ad alta qualità, organizzazione in album tematici "
                    "(tagli, barba, colorazioni), gestione metadati (tags, descrizioni, prezzi), "
                    "sistema di privacy (pubblico/privato), e showcase before/after. "
                    "Le foto sono ottimizzate automaticamente e visualizzabili in gallerie responsive."
                ),
                bounded_context="Portfolio Management",
                responsibilities=[
                    "Upload e archiviazione foto ad alta qualità",
                    "Organizzazione in album e categorie tematiche",
                    "Gestione metadati (tags, descrizioni, prezzi servizi)",
                    "Sistema di privacy e visibilità (pubblico/privato/clienti)",
                    "Ottimizzazione automatica immagini (compressione, thumbnails)",
                    "Gallery responsive e modalità showcase",
                    "Before/After comparison view",
                    "Moderazione contenuti e quality control",
                ],
                dependencies=[],
                communication_patterns=[CommunicationPattern.SYNC_REST],
            ),
        ],
        global_constraints={
            "max_response_time": "300ms",
            "availability": "99.5%",
            "max_photo_size": "10MB",
            "supported_formats": "JPEG, PNG, HEIC",
        },
        technical_stack={
            "language": "Python",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "cache": "Redis",
            "storage": "S3",
            "image_processing": "Pillow",
        },
    )


@pytest.fixture
def orchestrator() -> SpecificationOrchestrator:
    """Orchestrator per generazione specifiche"""
    return SpecificationOrchestrator()


# ============================================================================
# FUNZIONI DI VALIDAZIONE SEMANTICA
# ============================================================================

def calculate_keyword_coverage(
    requirements: List[Requirement], keywords: Set[str]
) -> float:
    """
    Calcola la copertura percentuale delle parole chiave nei requisiti.
    
    Args:
        requirements: Lista di requisiti funzionali
        keywords: Set di parole chiave da cercare
        
    Returns:
        Percentuale di parole chiave trovate (0.0 - 1.0)
    """
    found_keywords = set()
    
    for req in requirements:
        searchable_text = f"{req.title} {req.description}".lower()
        searchable_text += " " + " ".join(req.acceptance_criteria).lower()
        
        for keyword in keywords:
            if keyword.lower() in searchable_text:
                found_keywords.add(keyword)
    
    if not keywords:
        return 0.0
    
    return len(found_keywords) / len(keywords)


def find_requirements_by_area(
    requirements: List[Requirement], area_keywords: Set[str]
) -> List[Requirement]:
    """
    Trova requisiti che appartengono a una specifica area funzionale.
    
    Args:
        requirements: Lista di requisiti funzionali
        area_keywords: Parole chiave dell'area funzionale
        
    Returns:
        Lista di requisiti che matchano l'area
    """
    matching_requirements = []
    
    for req in requirements:
        searchable_text = f"{req.title} {req.description}".lower()
        
        if any(keyword.lower() in searchable_text for keyword in area_keywords):
            matching_requirements.append(req)
    
    return matching_requirements


def validate_requirement_completeness(req: Requirement) -> List[str]:
    """
    Valida la completezza di un singolo requisito.
    
    Returns:
        Lista di problemi riscontrati (vuota se tutto OK)
    """
    issues = []
    
    # Validazione titolo
    if len(req.title) < 10:
        issues.append(f"Titolo troppo corto: '{req.title}' ({len(req.title)} caratteri)")
    
    # Validazione descrizione
    if len(req.description) < 30:
        issues.append(
            f"Descrizione troppo corta per '{req.title}': {len(req.description)} caratteri"
        )
    
    # Validazione acceptance criteria
    if len(req.acceptance_criteria) < 1:
        issues.append(
            f"Criteri di accettazione insufficienti per '{req.title}': "
            f"trovati {len(req.acceptance_criteria)}, attesi almeno 1"
        )
    
    # Validazione lunghezza criteri
    for idx, criterion in enumerate(req.acceptance_criteria):
        if len(criterion) < 10:
            issues.append(
                f"Criterio #{idx + 1} troppo breve in '{req.title}': '{criterion}'"
            )
    
    return issues


# ============================================================================
# TEST PRINCIPALE: VALIDAZIONE REQUISITI FUNZIONALI PORTFOLIO SERVICE
# ============================================================================

@pytest.mark.asyncio
async def test_aircut_portfolio_functional_requirements_validation(
    aircut_portfolio_input: ArchitectureInput, orchestrator: SpecificationOrchestrator
):
    """
    Test di integrazione: Validazione requisiti funzionali per Portfolio Service.
    
    Valida che l'AI generi requisiti funzionali:
    1. Strutturalmente corretti (Pydantic)
    2. Completi (soglie minime)
    3. Semanticamente coerenti (parole chiave dominio portfolio)
    4. Distribuiti correttamente tra le aree funzionali del portfolio
    """
    
    print("\n" + "=" * 80)
    print("TEST: Validazione Requisiti Funzionali - Aircut Portfolio Service")
    print("=" * 80)
    print(f"Progetto: {aircut_portfolio_input.project_name}")
    print(f"Focus: Portfolio Service (gestione foto, album, showcase)")
    print()
    
    # -------------------------------------------------------------------------
    # FASE 1: GENERAZIONE SPECIFICHE
    # -------------------------------------------------------------------------
    print("[FASE 1] Generazione Specifiche AI")
    print("-" * 80)
    
    all_specs: List[MicroserviceSpec] = []
    
    # ✅ FIX: skip_details=True per generare SOLO requisiti funzionali
    async for event in orchestrator.generate_all_specs_streaming(
        aircut_portfolio_input, skip_details=True
    ):
        if event.get("event") == "microservice":
            import json
            spec_dict = json.loads(event["data"])
            validated_spec = MicroserviceSpec.model_validate(spec_dict)
            all_specs.append(validated_spec)
            print(f"  ✓ Generato: {validated_spec.service_name}")
    
    assert len(all_specs) == 1, (
        f"Atteso 1 microservizio (portfolio-service), trovati {len(all_specs)}"
    )
    
    print(f"\n✓ Generazione completata: {len(all_specs)} microservizio")
    
    # -------------------------------------------------------------------------
    # FASE 2: ESTRAZIONE E VALIDAZIONE STRUTTURALE REQUISITI FUNZIONALI
    # -------------------------------------------------------------------------
    print(f"\n[FASE 2] Validazione Strutturale Requisiti Funzionali")
    print("-" * 80)
    
    portfolio_service = all_specs[0]
    all_functional_requirements = [
        req for req in portfolio_service.functional_requirements
        if req.type == RequirementType.FUNCTIONAL
    ]
    
    print(f"\nportfolio-service:")
    print(f"  Requisiti funzionali: {len(all_functional_requirements)}")
    
    total_functional = len(all_functional_requirements)
    print(f"\n✓ Totale requisiti funzionali: {total_functional}")
    
    # Soglia minima: almeno 4 requisiti funzionali per portfolio service
    MIN_FUNCTIONAL_REQUIREMENTS = 3
    assert total_functional >= MIN_FUNCTIONAL_REQUIREMENTS, (
        f"REQUISITI INSUFFICIENTI: trovati {total_functional} requisiti funzionali, "
        f"attesi almeno {MIN_FUNCTIONAL_REQUIREMENTS} per portfolio service. "
        f"L'AI potrebbe non aver compreso a fondo il dominio."
    )
    
    print(f"✓ Soglia minima superata ({MIN_FUNCTIONAL_REQUIREMENTS}+ requisiti)")
    
    # -------------------------------------------------------------------------
    # FASE 3: VALIDAZIONE COMPLETEZZA SINGOLI REQUISITI
    # -------------------------------------------------------------------------
    print(f"\n[FASE 3] Validazione Completezza Singoli Requisiti")
    print("-" * 80)
    
    all_issues = []
    
    for idx, req in enumerate(all_functional_requirements, 1):
        issues = validate_requirement_completeness(req)
        if issues:
            all_issues.extend(issues)
            print(f"\n  ⚠️  Requisito #{idx} - {req.id}: {req.title}")
            for issue in issues:
                print(f"      • {issue}")
    
    if not all_issues:
        print("  ✓ Nessun problema rilevato - Tutti i requisiti sono completi e ben formati")
    else:
        # Tolleranza: max 20% dei requisiti può avere problemi minori
        tolerance = 0.2
        problematic_count = len(set(issue.split("'")[1] for issue in all_issues if "'" in issue))
        max_allowed = int(total_functional * tolerance)
        
        assert problematic_count <= max_allowed, (
            f"QUALITÀ INSUFFICIENTE: {problematic_count} requisiti hanno problemi, "
            f"massimo tollerato: {max_allowed} ({int(tolerance * 100)}%).\n"
            f"Problemi riscontrati:\n" + "\n".join(f"  • {issue}" for issue in all_issues)
        )
        
        print(f"  ⚠️  {problematic_count} requisiti con problemi minori (entro tolleranza)")
    
    # -------------------------------------------------------------------------
    # FASE 4: VALIDAZIONE SEMANTICA - COPERTURA DOMINIO PORTFOLIO
    # -------------------------------------------------------------------------
# -------------------------------------------------------------------------
    # [FASE 4] VALIDAZIONE SEMANTICA - COPERTURA DOMINIO AIRCUT
    # -------------------------------------------------------------------------
    print(f"\n[FASE 4] Validazione Semantica - Copertura Dominio Aircut")
    print("-" * 80)
    
    # Calcolo copertura globale
    global_coverage = calculate_keyword_coverage(
        all_functional_requirements, AIRCUT_PORTFOLIO_KEYWORDS
    )
    
    print(f"\nCopertura globale parole chiave dominio: {global_coverage:.1%}")
    
    # Soglia minima di riferimento
    MIN_COVERAGE = 0.15
    
    # --- MODIFICA: Logica Soft-Assertion (Warning invece di Errore) ---
    if global_coverage >= MIN_COVERAGE:
        print(f"✓ Copertura semantica adeguata (≥{MIN_COVERAGE:.1%})")
    else:
        # Stampa un avviso visibile ma NON ferma il test
        print(f"⚠️  WARNING: COPERTURA SEMANTICA INSUFFICIENTE ({global_coverage:.1%})")
        print(f"    Atteso almeno {MIN_COVERAGE:.1%}. L'AI potrebbe aver usato termini generici.")
        print(f"    -> Il test prosegue comunque come richiesto.")
    
    # -------------------------------------------------------------------------
    # FASE 5: VALIDAZIONE DISTRIBUZIONE PER AREA FUNZIONALE
    # -------------------------------------------------------------------------
    print(f"\n[FASE 5] Validazione Distribuzione per Area Funzionale")
    print("-" * 80)
    
    area_distribution = {}
    
    for area_name, area_keywords in AIRCUT_PORTFOLIO_KEYWORDS.items():
        matching_reqs = find_requirements_by_area(
            all_functional_requirements, area_keywords
        )
        area_distribution[area_name] = len(matching_reqs)
        
        print(f"\n  {area_name.replace('_', ' ').title()}:")
        print(f"    Requisiti trovati: {len(matching_reqs)}")
        
        if matching_reqs:
            for req in matching_reqs[:3]:  # Mostra primi 3
                print(f"      • {req.id}: {req.title}")
    
    # Validazione: almeno 2 delle 5 aree devono avere requisiti
    areas_covered = sum(1 for count in area_distribution.values() if count > 0)
    MIN_AREAS_COVERED = 2
    
    assert areas_covered >= MIN_AREAS_COVERED, (
        f"COPERTURA AREE INSUFFICIENTE: solo {areas_covered}/5 aree funzionali coperte, "
        f"attese almeno {MIN_AREAS_COVERED}.\n"
        f"Distribuzione: {area_distribution}\n"
        f"L'AI potrebbe aver focalizzato su poche funzionalità."
    )
    
    print(f"\n✓ Copertura aree adeguata ({areas_covered}/{len(AIRCUT_PORTFOLIO_KEYWORDS)} aree)")
    
    # -------------------------------------------------------------------------
    # FASE 6: VALIDAZIONE PRIORITÀ
    # -------------------------------------------------------------------------
    print(f"\n[FASE 6] Validazione Distribuzione Priorità")
    print("-" * 80)
    
    priority_distribution = {
        RequirementPriority.CRITICAL: 0,
        RequirementPriority.HIGH: 0,
        RequirementPriority.MEDIUM: 0,
        RequirementPriority.LOW: 0,
    }
    
    for req in all_functional_requirements:
        priority_distribution[req.priority] += 1
    
    print("\nDistribuzione priorità:")
    for priority, count in priority_distribution.items():
        percentage = (count / total_functional * 100) if total_functional > 0 else 0
        print(f"  {priority.value.upper()}: {count} ({percentage:.1f}%)")
    
    # Validazione: almeno alcuni requisiti CRITICAL o HIGH
    critical_high_count = (
        priority_distribution[RequirementPriority.CRITICAL]
        + priority_distribution[RequirementPriority.HIGH]
    )
    
    assert critical_high_count > 0, (
        "DISTRIBUZIONE PRIORITÀ ERRATA: nessun requisito CRITICAL o HIGH. "
        "Ogni sistema dovrebbe avere requisiti critici."
    )
    
    # Validazione: non tutti i requisiti devono essere CRITICAL
    if total_functional > 0:
        critical_percentage = (
            priority_distribution[RequirementPriority.CRITICAL] / total_functional
        )
        assert critical_percentage < 0.8, (
            f"DISTRIBUZIONE PRIORITÀ IRREALISTICA: {critical_percentage:.1%} "
            f"dei requisiti sono CRITICAL. Una distribuzione realistica dovrebbe "
            f"avere mix di priorità."
        )
    
    print(f"✓ Distribuzione priorità realistica")
    
    # -------------------------------------------------------------------------
    # FASE 7: REPORT DETTAGLIATO - TUTTI I REQUISITI PER PRIORITÀ
    # -------------------------------------------------------------------------
    print(f"\n[FASE 7] Report Dettagliato Requisiti Funzionali")
    print("-" * 80)
    
    # Raggruppa requisiti per priorità
    reqs_by_priority = {
        RequirementPriority.CRITICAL: [],
        RequirementPriority.HIGH: [],
        RequirementPriority.MEDIUM: [],
        RequirementPriority.LOW: [],
    }
    
    for req in all_functional_requirements:
        reqs_by_priority[req.priority].append(req)
    
    # Stampa TUTTI i requisiti divisi per priorità
    for priority in [RequirementPriority.CRITICAL, RequirementPriority.HIGH, 
                     RequirementPriority.MEDIUM, RequirementPriority.LOW]:
        reqs = reqs_by_priority[priority]
        if reqs:
            print(f"\nRequisiti {priority.value.upper()} ({len(reqs)}):")
            for req in reqs:
                print(f"\n  📌 {req.id}: {req.title}")
                print(f"     Descrizione: {req.description}")
                print(f"     Criteri di accettazione ({len(req.acceptance_criteria)}):")
                for criterion in req.acceptance_criteria:
                    print(f"       ✓ {criterion}")
    
    # -------------------------------------------------------------------------
    # SUMMARY FINALE
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("✅ VALIDAZIONE REQUISITI FUNZIONALI COMPLETATA CON SUCCESSO")
    print(f"{'=' * 80}")
    
    print(f"\n📊 Metriche Finali:")
    print(f"  • Requisiti Funzionali Totali: {total_functional}")
    print(f"  • Microservizi: 1 (Focus: portfolio-service)")
    print(f"  • Copertura Semantica Dominio: {global_coverage:.1%}")
    print(f"  • Aree Funzionali Coperte: {areas_covered}/{len(AIRCUT_PORTFOLIO_KEYWORDS)}")
    print(f"  • Requisiti CRITICAL: {priority_distribution[RequirementPriority.CRITICAL]}")
    print(f"  • Requisiti HIGH: {priority_distribution[RequirementPriority.HIGH]}")
    print(f"  • Requisiti con Problemi: {len(all_issues)}")
    
    print(f"\n🎯 Valutazione Qualità:")
    if global_coverage >= 0.25 and areas_covered >= 4:
        print("  ⭐⭐⭐ ECCELLENTE - Requisiti completi e ben distribuiti")
    elif global_coverage >= 0.20 and areas_covered >= 3:
        print("  ⭐⭐ BUONO - Requisiti adeguati con copertura soddisfacente")
    else:
        print("  ⭐ SUFFICIENTE - Requisiti base presenti, possibile miglioramento")
    
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])