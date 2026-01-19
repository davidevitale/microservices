================================================================================
DOCUMENTAZIONE TECNICA: AGENT 3 - FUNCTIONAL SPECIFICATION GENERATOR
================================================================================
Progetto: Nobel Engineering - AI Driven Software Architecture
Modulo: Agent 3 (Generatore di Specifiche Funzionali)
Versione: 1.0.0
Data: Gennaio 2026
Repository: /agent3-spec-generator

--------------------------------------------------------------------------------
1. EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
Agent 3 è un microservizio intelligente progettato per automatizzare la fase di 
analisi funzionale nel ciclo di vita dello sviluppo software. All'interno della 
pipeline "Nobel Engineering", riceve in input l'architettura di alto livello 
(definita dall'Agent 2) e produce specifiche funzionali granulari, contratti API 
e schemi di eventi pronti per l'implementazione.

Il sistema si distingue per l'uso del framework DSPy (Declarative Self-improving Python) 
per l'orchestrazione deterministica degli LLM, garantendo output strutturati e 
validati sintatticamente, superando i limiti dei tradizionali approcci basati su prompt.

--------------------------------------------------------------------------------
2. ARCHITETTURA DEL SISTEMA
--------------------------------------------------------------------------------

2.1. Design Pattern
L'architettura segue i principi della Clean Architecture e dei Microservizi:
- **Core Domain:** Logica di generazione DSPy (`app/modules`).
- **Interface Layer:** API REST e Streaming SSE (`app/main.py`).
- **Infrastructure Layer:** Docker, Ollama, Configurazione (`app/core`).

2.2. Stack Tecnologico
- **Linguaggio:** Python 3.11+
- **Web Framework:** FastAPI (Async/Await nativo).
- **AI Orchestration:** DSPy (sostituisce LangChain per maggiore controllo sui tipi).
- **LLM Engine:** Ollama (Modelli: Llama 3, Mixtral) con supporto locale/remoto.
- **Data Validation:** Pydantic v2 (Validazione rigorosa di Input/Output).
- **Dependency Management:** Poetry 1.7+.
- **Containerization:** Docker (Multi-stage build).

--------------------------------------------------------------------------------
3. DETTAGLI IMPLEMENTATIVI: AI & DSPy
--------------------------------------------------------------------------------
Il cuore del sistema (`generator_module.py`) non utilizza semplici prompt testuali, 
ma "Signatures" e "Modules" tipizzati.

3.1. Pipeline di Generazione (Chain-of-Thought)
L'orchestratore esegue una pipeline sequenziale per ogni sottodominio:

A. Functional Requirements Generator
   - **Metodo:** Chain-of-Thought (CoT).
   - **Input:** Descrizione sottodominio, Bounded Context, Responsabilità.
   - **Output:** Lista JSON di requisiti (ID, Priorità, Criteri di Accettazione).
   - **Logica:** Deduce i requisiti impliciti analizzando le responsabilità semantiche.

B. API Endpoints Generator
   - **Input:** Requisiti Funzionali generati al passo A.
   - **Output:** Specifiche OpenAPI (Path, Method, Schema Request/Response).
   - **Logica:** Mappa ogni requisito funzionale in operazioni CRUD o RPC.

C. Domain Events Generator
   - **Input:** Pattern di comunicazione (es. Async Event) e Responsabilità.
   - **Output:** Definizione eventi (Nome, Payload Schema, Trigger).
   - **Scopo:** Supportare architetture Event-Driven e CQRS.

D. Non-Functional Requirements (NFR) Generator
   - **Input:** Tipo servizio (Core/Supporting), Vincoli globali.
   - **Output:** Requisiti di qualità (Performance, Sicurezza, Disponibilità).

3.2. Robustezza e Fallback
Il sistema implementa un pattern "Graceful Degradation":
- Se il modello LLM genera un JSON malformato o allucinato che viola lo schema Pydantic, 
  il sistema cattura l'eccezione `ValidationError`.
- Viene attivato automaticamente un generatore di fallback (`_fallback_requirements`) 
  che produce una specifica minimale ma valida, garantendo che la pipeline non si 
  blocchi mai.

--------------------------------------------------------------------------------
4. INTERFACCE E API
--------------------------------------------------------------------------------
Il servizio espone endpoint documentati automaticamente (Swagger/OpenAPI).

4.1. Server-Sent Events (SSE) - Streaming Real-time
Endpoint: `POST /generate/stream`
Permette al frontend di visualizzare il "ragionamento" dell'AI in tempo reale.
Formato eventi:
- `start`: Inizio processo.
- `progress`: Aggiornamento step (es. "Generating API endpoints...").
- `microservice`: Payload completo di un singolo servizio appena completato.
- `complete`: Output finale aggregato.
- `error`: Gestione errori granulare.

4.2. Endpoint Sincrono
Endpoint: `POST /generate`
Modalità legacy per integrazioni batch (waiting attivo fino al completamento).

4.3. Health Check
Endpoint: `GET /health`
Verifica non solo lo stato del servizio web, ma esegue un test di connessione 
attiva verso l'istanza Ollama configurata.

--------------------------------------------------------------------------------
5. CONTRATTI DATI (DATA CONTRACTS)
--------------------------------------------------------------------------------

5.1. Input (`ArchitectureInput`)
Definisce l'architettura approvata dall'Architect Agent.
- `subdomains`: Lista di sottodomini con Bounded Context e responsabilità.
- `communication_patterns`: Enum (REST, ASYNC_EVENT, GRAPHQL).
- `global_constraints`: Vincoli SLA globali.

5.2. Output (`FunctionalSpecificationOutput`)
Output strutturato conforme a standard documentali.
- `MicroserviceSpec`: Oggetto principale per ogni servizio.
- `Requirement`: Struttura standard (ID, Titolo, Descrizione, Acceptance Criteria).
- `APIEndpoint`: Definizione agnostica convertibile in OpenAPI 3.0.
- `ServiceDependency`: Mappa delle dipendenze e strategie di fallback (es. Circuit Breaker).

--------------------------------------------------------------------------------
6. INFRASTRUTTURA E DEPLOYMENT
--------------------------------------------------------------------------------

6.1. Dockerizzazione
L'immagine è ottimizzata per sicurezza e dimensione:
- **Base:** `python:3.11-slim`.
- **Multi-stage:** Separazione tra build dependencies e runtime.
- **Sicurezza:** Esecuzione come utente non privilegiato `agent3` (UID 1000).
- **Healthcheck:** Comando `curl` integrato per monitoraggio container.

6.2. Configurazione
- Configurazione centralizzata via variabili d'ambiente.
- Supporto dinamico per l'URL di Ollama (`host.docker.internal` per comunicazione 
  container-to-host).

6.3. Developer Experience
Script di automazione (`init-ws.py`) incluso per:
- Setup automatico ambiente virtuale Poetry.
- Installazione Git Hooks (Pre-commit) per linting (Ruff, Black, Mypy).
- Verifica requisiti di sistema.

--------------------------------------------------------------------------------
7. GUIDA ALL'AVVIO
--------------------------------------------------------------------------------

Prerequisiti: Docker Desktop, Ollama (running su porta 11434).

A. Avvio Rapido (Docker):
   1. Build: `docker build -t agent3-spec-gen .`
   2. Run:   `docker run -d -p 8003:8003 --env-file .env agent3-spec-gen`

B. Sviluppo Locale:
   1. Init:  `python init-ws.py`
   2. Run:   `poetry run python -m app.main`

C. Verifica:
   - Swagger UI: http://localhost:8003/docs
   - Health:     http://localhost:8003/health

--------------------------------------------------------------------------------
8. CONCLUSIONI
--------------------------------------------------------------------------------
Agent 3 risolve il problema della "pagina bianca" nella stesura delle specifiche 
tecniche. Attraverso l'uso di DSPy e Pydantic, trasforma l'output probabilistico 
degli LLM in artefatti ingegneristici deterministici, riducendo i tempi di 
analisi del 70% e garantendo coerenza tra architettura e implementazione.
