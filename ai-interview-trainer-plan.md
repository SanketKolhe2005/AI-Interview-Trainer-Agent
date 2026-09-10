# AI Interview Trainer Agent — Implementation Plan

## Overview

**Goal:** Build a complete AI Interview Trainer Agent powered by RAG and IBM Granite LLM that prepares users for job interviews by generating tailored question sets, model answers, improvement tips, preparation strategies, and an interactive mock interview session.

**Scope:** Greenfield Python project. No existing code. Must satisfy AICTE IBM SkillsBuild Problem Statement No. 22.

**Non-Goals:**
- No user authentication / login system
- No persistent database for user sessions
- No deployment to production cloud (IBM Cloud Lite free tier only)
- No mobile app

**Mandatory Technologies:**
- IBM watsonx.ai (IBM Cloud Lite) — for IBM Granite LLM inference
- IBM Granite model (`ibm/granite-13b-chat-v2` or equivalent available on Lite tier)
- RAG (LangChain + ChromaDB)
- Gradio (UI)
- Python 3.10+

---

## Architecture Summary

### Layers
1. **UI Layer** — Gradio web interface (profile input, output tabs, mock interview)
2. **Agent Layer** — Orchestrator + 6 specialized generation modules
3. **RAG Layer** — Document loader → chunker → embedder → ChromaDB → retriever
4. **LLM Layer** — IBM Granite via watsonx.ai Python SDK
5. **Knowledge Base** — Curated JSON/TXT files per role and question type
6. **Config/Utils** — Environment config, profile parser, output formatter

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffolding

**Intent:** Establish the full project directory structure, dependency file, environment config template, and `.gitignore` so all subsequent sub-tasks have a clean, consistent foundation.

**Expected Outcomes:**
- All directories exist as specified in the file structure
- `requirements.txt` lists all Python dependencies with pinned versions
- `.env.example` documents all required environment variables
- `.gitignore` excludes secrets, vector store data, and Python cache
- `config/settings.py` centralises all app-wide constants

**Todo List:**
1. Create all directories: `agent/`, `rag/`, `llm/`, `knowledge_base/` (with subdirs), `prompts/`, `utils/`, `config/`, `tests/`
2. Create `requirements.txt` with: `ibm-watsonx-ai`, `langchain`, `langchain-community`, `chromadb`, `sentence-transformers`, `gradio`, `python-dotenv`, `pypdf`, `tiktoken`, `pytest`
3. Create `.env.example` with: `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, `GRANITE_MODEL_ID`, `EMBEDDING_MODEL`, `CHROMA_PERSIST_DIR`, `TOP_K_RETRIEVAL`
4. Create `.gitignore` — exclude `.env`, `__pycache__`, `*.pyc`, `chroma_db/`, `.venv/`, `*.egg-info`
5. Create `config/settings.py` — load `.env` and expose typed constants
6. Create all `__init__.py` files in each Python package directory
7. Create `README.md` placeholder with project title, domain, and problem statement

**Relevant Context:**
- IBM watsonx.ai Python SDK package: `ibm-watsonx-ai`
- ChromaDB is used as local persistent vector store
- All secrets must be in `.env`, never hardcoded

**Status:** [x] done

---

### Sub-Task 2 — Knowledge Base Creation

**Intent:** Populate the `knowledge_base/` directory with curated, structured documents that the RAG pipeline will index and retrieve from. These documents are the ground truth for all question generation.

**Expected Outcomes:**
- At minimum 5 role-specific technical question banks (e.g. Software Engineer, Data Scientist, DevOps, Product Manager, Business Analyst) in JSON format
- HR question bank (JSON) with at least 20 standard HR questions
- Behavioral question bank (JSON) with at least 15 STAR-format scenarios
- Role description summaries (TXT) for each supported role
- Interview preparation guidelines document (TXT)

**Todo List:**
1. Create `knowledge_base/technical_questions/` — one JSON file per role (software_engineer.json, data_scientist.json, devops_engineer.json, product_manager.json, business_analyst.json). Each entry: `{question, difficulty, topic, role}`
2. Create `knowledge_base/hr_questions/hr_questions.json` — entries: `{question, category, intent}`
3. Create `knowledge_base/behavioral_questions/behavioral_questions.json` — entries: `{question, competency, star_hint}`
4. Create `knowledge_base/role_descriptions/` — one TXT per role summarising responsibilities, required skills, and common interview focus areas
5. Create `knowledge_base/industry_guidelines/interview_prep_guide.txt` — general interview preparation strategies, body language tips, STAR method explanation, common mistakes

**Relevant Context:**
- JSON is preferred over PDF for structured Q banks (faster to parse, no OCR needed)
- TXT is fine for narrative guides
- All content should be factual and role-relevant, not generic filler

**Status:** [x] done

---

### Sub-Task 3 — RAG Pipeline

**Intent:** Build the full retrieval pipeline: document loader, text chunker, embedding generator, ChromaDB vector store manager, and semantic retriever. This is the core intelligence layer that connects knowledge documents to the LLM.

**Expected Outcomes:**
- `rag/loader.py` can load all knowledge base documents (JSON + TXT) into LangChain `Document` objects
- `rag/chunker.py` splits documents into appropriately sized chunks with metadata preserved
- `rag/embedder.py` generates embeddings using a sentence-transformers model (offline, no API cost)
- `rag/vector_store.py` persists and loads a ChromaDB collection
- `rag/retriever.py` performs top-K semantic similarity search given a user query string
- A standalone `rag/ingest.py` script can be run once to build the vector store from scratch

**Todo List:**
1. Implement `rag/loader.py` — `load_json_documents(path)` and `load_text_documents(path)` functions returning `List[Document]` with metadata (role, category, source)
2. Implement `rag/chunker.py` — `chunk_documents(docs, chunk_size=500, overlap=50)` using LangChain `RecursiveCharacterTextSplitter`
3. Implement `rag/embedder.py` — wrapper around `sentence-transformers/all-MiniLM-L6-v2` compatible with LangChain `Embeddings` interface
4. Implement `rag/vector_store.py` — `build_store(chunks)` and `load_store()` using ChromaDB with persistence to `CHROMA_PERSIST_DIR`
5. Implement `rag/retriever.py` — `retrieve(query, role_filter=None, k=TOP_K_RETRIEVAL)` returning top-K relevant chunks
6. Implement `rag/ingest.py` — orchestrates load → chunk → embed → store; runnable as `python -m rag.ingest`

**Relevant Context:**
- Use `langchain_community.document_loaders` for document loading
- Use `langchain_community.vectorstores.Chroma` for vector store
- `sentence-transformers/all-MiniLM-L6-v2` is free, fast, and works offline — no IBM quota consumed for embeddings
- Metadata filtering by `role` allows more precise retrieval per user profile

**Status:** [x] done

---

### Sub-Task 4 — IBM Granite LLM Client

**Intent:** Implement the `llm/granite_client.py` module that wraps IBM watsonx.ai SDK calls to IBM Granite, providing a clean `generate(prompt)` interface used by all agent modules.

**Expected Outcomes:**
- `GraniteClient` class initialises from environment variables
- `generate(prompt, max_tokens, temperature)` sends a prompt to IBM Granite and returns the text response
- Graceful error handling for API errors, quota limits, and timeouts
- The client is compatible with LangChain `LLM` interface (optional but preferred)

**Todo List:**
1. Implement `llm/granite_client.py` — `GraniteClient` class with `__init__` loading `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, `GRANITE_MODEL_ID` from settings
2. Implement `generate(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9)` method using `ibm_watsonx_ai.foundation_models.ModelInference`
3. Add LangChain-compatible wrapper so the client can be used as `llm=` parameter in LangChain chains
4. Add a simple `test_connection()` method that sends a minimal prompt to verify credentials work

**Relevant Context:**
- IBM watsonx.ai SDK: `from ibm_watsonx_ai import Credentials` and `from ibm_watsonx_ai.foundation_models import ModelInference`
- Granite model IDs available on Cloud Lite: `ibm/granite-13b-chat-v2`, `ibm/granite-3-8b-instruct` — make model ID configurable via `.env`
- watsonx.ai URL for Dallas region: `https://us-south.ml.cloud.ibm.com`

**Status:** [x] done

---

### Sub-Task 5 — Prompt Templates

**Intent:** Create well-engineered prompt templates for each of the 7 output types. Prompts must be structured, role-aware, and produce consistent, parseable output from IBM Granite.

**Expected Outcomes:**
- 7 prompt template files in `prompts/` directory
- Each template uses `{placeholders}` for dynamic injection of role, experience level, skills, and retrieved context
- Prompts instruct the model to return structured output (numbered lists, JSON blocks, or clearly labelled sections)

**Todo List:**
1. Create `prompts/technical_prompt.txt` — instructs Granite to generate 5-10 technical questions given role, experience level, and retrieved technical context
2. Create `prompts/role_specific_prompt.txt` — generates role-specific scenario questions based on JD and retrieved role descriptions
3. Create `prompts/hr_prompt.txt` — generates 5 HR questions personalised to user background
4. Create `prompts/behavioral_prompt.txt` — generates 5 behavioral STAR-format questions based on skills and retrieved behavioral context
5. Create `prompts/answer_prompt.txt` — generates a model answer for a given question, role, and experience level
6. Create `prompts/tips_prompt.txt` — generates improvement tips based on user profile and question set
7. Create `prompts/strategy_prompt.txt` — generates a 7-day interview preparation strategy
8. Create `prompts/mock_interview_prompt.txt` — evaluates a user's answer and provides structured feedback

**Relevant Context:**
- IBM Granite responds best to clear instruction-following prompts with explicit output format instructions
- Use `<context>`, `<question>`, `<answer>` XML-style delimiters as Granite is trained on instruction-following formats
- Keep prompts under 2000 tokens to leave headroom for responses within token limits

**Status:** [x] done

---

### Sub-Task 6 — Agent Modules

**Intent:** Implement the 6 specialized generation modules and the central orchestrator that routes user profile data through the RAG pipeline and LLM to produce all required outputs.

**Expected Outcomes:**
- `agent/orchestrator.py` — takes `UserProfile` and returns a complete `InterviewKit` object
- `agent/question_generator.py` — generates all 4 question types (technical, role-specific, HR, behavioral)
- `agent/answer_generator.py` — generates model answers for each question
- `agent/tips_generator.py` — generates improvement tips
- `agent/strategy_generator.py` — generates preparation strategy
- `agent/mock_interview.py` — manages stateful mock interview conversation turns
- `utils/profile_parser.py` — parses and validates user input into a `UserProfile` dataclass
- `utils/formatter.py` — formats agent output into display-ready strings for the UI

**Todo List:**
1. Define `UserProfile` dataclass in `utils/profile_parser.py` — fields: `name`, `role`, `experience_level`, `skills` (list), `resume_summary`
2. Define `InterviewKit` dataclass in `agent/orchestrator.py` — fields: `technical_questions`, `role_questions`, `hr_questions`, `behavioral_questions`, `model_answers`, `tips`, `strategy`
3. Implement `agent/question_generator.py` — `QuestionGenerator` class with `generate_technical()`, `generate_role_specific()`, `generate_hr()`, `generate_behavioral()` methods; each retrieves from RAG then calls Granite
4. Implement `agent/answer_generator.py` — `AnswerGenerator` class with `generate_answers(questions, profile)` method
5. Implement `agent/tips_generator.py` — `TipsGenerator` class with `generate_tips(profile, questions)` method
6. Implement `agent/strategy_generator.py` — `StrategyGenerator` class with `generate_strategy(profile)` method
7. Implement `agent/mock_interview.py` — `MockInterviewer` class maintaining conversation history; `start_session()`, `ask_question()`, `evaluate_answer(user_answer)`, `get_next_question()` methods
8. Implement `agent/orchestrator.py` — `InterviewOrchestrator` that composes all modules and returns full `InterviewKit`
9. Implement `utils/formatter.py` — functions to render `InterviewKit` sections as markdown strings for Gradio

**Relevant Context:**
- Each generator follows: retrieve context from RAG → inject into prompt template → call Granite → parse response
- Mock interview is stateful — maintain a list of questions and track current index
- Use Python dataclasses (`@dataclass`) for `UserProfile` and `InterviewKit` for clean, typed interfaces

**Status:** [x] done

---

### Sub-Task 7 — Gradio UI

**Intent:** Build the Gradio web interface that allows users to input their profile, view all generated outputs in organised tabs, and conduct an interactive mock interview session.

**Expected Outcomes:**
- `app.py` launches a Gradio app accessible on localhost
- Tab 1: Profile Input — form fields for name, role (dropdown), experience level (dropdown), skills (text), resume summary (text area)
- Tab 2: Interview Questions — 4 sub-sections for each question type
- Tab 3: Model Answers — question + answer pairs
- Tab 4: Tips & Strategy — improvement tips and 7-day preparation plan
- Tab 5: Mock Interview — chat interface for interactive Q&A with AI feedback

**Todo List:**
1. Implement `app.py` — define Gradio `Blocks` layout with 5 tabs
2. Implement profile input form in Tab 1 with a "Generate Interview Kit" button
3. Implement Tab 2 display — output boxes for technical, role-specific, HR, and behavioral questions
4. Implement Tab 3 display — model answers rendered as markdown accordion or text
5. Implement Tab 4 display — tips and strategy as formatted markdown
6. Implement Tab 5 — `gr.ChatInterface` connected to `MockInterviewer.evaluate_answer()` with session state
7. Wire the "Generate" button to `InterviewOrchestrator.run(profile)` and populate all output tabs
8. Add loading spinner / progress indicator during generation
9. Add a "Reset" button that clears all outputs and session state

**Relevant Context:**
- Gradio `gr.Blocks` with `gr.Tab` components is the right pattern for multi-tab layout
- `gr.State` is used to maintain `MockInterviewer` instance across chat turns
- Role options dropdown should match the roles present in the knowledge base

**Status:** [x] done

---

### Sub-Task 8 — End-to-End Integration

**Intent:** Wire all layers together, verify the full user journey works from profile input to mock interview output, and resolve any integration issues between components.

**Expected Outcomes:**
- Running `python app.py` launches the full working application
- A complete end-to-end flow works: profile input → RAG retrieval → Granite generation → structured output display → mock interview
- `rag/ingest.py` runs successfully and builds the vector store
- No import errors, missing files, or broken references

**Todo List:**
1. Run `python -m rag.ingest` and verify vector store is created in `CHROMA_PERSIST_DIR`
2. Run `python -c "from llm.granite_client import GraniteClient; GraniteClient().test_connection()"` to verify API connectivity
3. Run `python app.py` and test the complete UI flow with a sample profile
4. Verify all 5 tabs render correctly and all outputs are populated
5. Verify mock interview chat responds with question, accepts answer, and returns feedback
6. Fix any import errors, prompt formatting issues, or missing context errors found during integration

**Relevant Context:**
- The vector store must be built (ingest step) before the app can serve retrieval requests
- API key and project ID must be set in `.env` before testing the LLM layer
- Integration testing is manual at this stage (automated tests are in Sub-Task 9)

**Status:** [x] done

---

### Sub-Task 9 — Testing

**Intent:** Write automated tests for the RAG pipeline, agent modules, and LLM client to ensure correctness and catch regressions.

**Expected Outcomes:**
- `tests/test_rag.py` — tests for loader, chunker, and retriever with mock documents
- `tests/test_agent.py` — tests for question generator and orchestrator with mocked LLM responses
- `tests/test_llm.py` — tests for `GraniteClient` with mocked API calls
- All tests pass with `pytest tests/`

**Todo List:**
1. Write `tests/test_rag.py` — test `load_json_documents`, `chunk_documents`, and `retrieve` with local fixture documents
2. Write `tests/test_agent.py` — test `QuestionGenerator` and `InterviewOrchestrator` with `unittest.mock.patch` on `GraniteClient.generate`
3. Write `tests/test_llm.py` — test `GraniteClient.__init__` config loading and mock the `ModelInference` API call
4. Write `tests/test_ui.py` — test `profile_parser` validation and `formatter` output formatting
5. Run `pytest tests/ -v` and fix any failures

**Relevant Context:**
- Use `pytest` and `unittest.mock` — no additional test framework needed
- LLM API calls must be mocked in tests to avoid consuming IBM Cloud quota during CI
- Keep fixture documents small (3-5 entries) to keep tests fast

**Status:** [x] done

---

### Sub-Task 10 — Documentation

**Intent:** Write a comprehensive README and supporting documentation so the project is GitHub-ready, reproducible, and clearly explains setup, architecture, and usage.

**Expected Outcomes:**
- `README.md` — complete with badges, project description, architecture overview, setup instructions, usage guide, and screenshots placeholder
- `README.md` includes the AICTE IBM SkillsBuild Problem Statement reference
- `.env.example` is documented inline
- All code files have module-level docstrings

**Todo List:**
1. Write full `README.md` with sections: Project Title, Problem Statement, Features, Architecture, Tech Stack, Prerequisites, Setup & Installation, Running the App, Running Tests, Project Structure, Team/Credits
2. Add inline comments to `.env.example` explaining each variable
3. Add module-level docstrings to all Python files
4. Add a `CONTRIBUTING.md` with basic contribution guidelines
5. Verify the README setup instructions reproduce a working install from scratch

**Relevant Context:**
- README should be written for a reviewer / evaluator who is familiar with Python but not the codebase
- Include the exact `pip install -r requirements.txt` and `python -m rag.ingest` commands
- Mention IBM Cloud Lite account requirement and how to obtain `WATSONX_API_KEY`

**Status:** [x] done

---

## File Structure Reference

```
AI-Interview-Trainer-Agent/
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── question_generator.py
│   ├── answer_generator.py
│   ├── tips_generator.py
│   ├── strategy_generator.py
│   └── mock_interview.py
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── ingest.py
├── llm/
│   ├── __init__.py
│   └── granite_client.py
├── knowledge_base/
│   ├── technical_questions/
│   │   ├── software_engineer.json
│   │   ├── data_scientist.json
│   │   ├── devops_engineer.json
│   │   ├── product_manager.json
│   │   └── business_analyst.json
│   ├── hr_questions/
│   │   └── hr_questions.json
│   ├── behavioral_questions/
│   │   └── behavioral_questions.json
│   ├── role_descriptions/
│   │   ├── software_engineer.txt
│   │   ├── data_scientist.txt
│   │   ├── devops_engineer.txt
│   │   ├── product_manager.txt
│   │   └── business_analyst.txt
│   └── industry_guidelines/
│       └── interview_prep_guide.txt
├── prompts/
│   ├── technical_prompt.txt
│   ├── role_specific_prompt.txt
│   ├── hr_prompt.txt
│   ├── behavioral_prompt.txt
│   ├── answer_prompt.txt
│   ├── tips_prompt.txt
│   ├── strategy_prompt.txt
│   └── mock_interview_prompt.txt
├── utils/
│   ├── __init__.py
│   ├── profile_parser.py
│   └── formatter.py
└── tests/
    ├── test_rag.py
    ├── test_agent.py
    ├── test_llm.py
    └── test_ui.py
```

---

## Technology Stack Summary

| Component | Technology | Reason |
|---|---|---|
| LLM | IBM Granite via watsonx.ai | Mandatory per problem statement |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free, offline, no quota cost |
| Vector Store | ChromaDB | Local persistence, no infra needed for Lite tier |
| RAG Framework | LangChain | Industry standard, works with ChromaDB + watsonx.ai |
| UI | Gradio | Fast, Python-native, good for demo/evaluation |
| Cloud | IBM Cloud Lite | Mandatory per problem statement |
| Language | Python 3.10+ | Ecosystem compatibility |

---

## Open Questions / Decisions Made

1. **Embeddings:** Using `sentence-transformers` locally (not IBM Granite embeddings) to conserve IBM Cloud Lite token quota for generation only. Can be switched to IBM embedding models if quota allows.
2. **Vector Store:** ChromaDB (local) instead of IBM Watson Discovery to stay within IBM Cloud Lite free tier constraints.
3. **5 Roles Supported Initially:** Software Engineer, Data Scientist, DevOps Engineer, Product Manager, Business Analyst. Additional roles can be added by dropping JSON files in `knowledge_base/technical_questions/`.
4. **Mock Interview:** Stateful within a single Gradio session (not persisted). No user accounts required.
