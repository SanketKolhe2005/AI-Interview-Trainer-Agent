🎯 AI Interview Trainer Agent

AICTE IBM SkillsBuild — Problem Statement No. 22 — Interview Trainer Agent

Domain: Education / Career Development






📋 Problem Statement

An Interview Trainer Agent, powered by Retrieval-Augmented Generation (RAG), prepares users for job interviews by generating tailored question sets and preparation strategies based on their profile, experience level, and job role.

The agent retrieves role-specific interview questions, industry expectations, behavioral scenarios, and HR guidelines from a curated knowledge base, and uses IBM Granite via IBM watsonx.ai to generate personalized interview content.

The application also provides a Local Fallback Mode so that interview preparation can continue when IBM Granite is temporarily unavailable because of quota or service limitations.

🎯 Project Objectives

Help candidates prepare effectively for professional job interviews across multiple roles and experience levels.

Generate role-specific and experience-calibrated interview questions using RAG and IBM Granite.

Evaluate candidate answers and provide structured, actionable feedback.

Conduct interactive mock interviews that simulate real interview conditions.

Produce a personalized 7-day interview preparation strategy.

Demonstrate a complete RAG-based AI application using IBM watsonx.ai and IBM Granite.

✨ Key Features

Feature

Description

Resume / Profile Analysis

Parses candidate name, role, experience, skills, and resume summary into a structured profile

Technical Questions

Generates technical interview questions based on role, experience, and skills

Role-Specific Questions

Generates questions aligned with the responsibilities of the selected job role

HR Questions

Generates personalized HR and career-related interview questions

Behavioral Questions

Generates behavioral questions based on common interview competencies and STAR methodology

RAG-Based Retrieval

Retrieves relevant context from the curated knowledge base using ChromaDB

IBM Granite Generation

Uses IBM Granite through IBM watsonx.ai for AI-powered content generation

Model Answers

Provides interview-ready answers for generated questions

Answer Evaluation

Evaluates candidate answers with scores, strengths, weaknesses, and improvement suggestions

Personalized Feedback

Provides structured interview feedback and preparation recommendations

Interactive Mock Interview

Conducts a stateful interview one question at a time with feedback

Final Feedback

Generates an overall performance report and 7-day preparation plan

Local Fallback Mode

Keeps the application functional when IBM Granite is unavailable

Gradio Interface

Provides an interactive 7-tab web interface

🏗️ Architecture Overview

Gradio Web UI
      │
      ▼
Interview Orchestrator
      │
      ├───────────────┐
      ▼               ▼
RAG Pipeline      IBM Granite LLM
      │               │
      ▼               │
Knowledge Base        │
      │               │
      └───────┬───────┘
              ▼
      Local Fallback Mode

🔄 Complete Project Workflow

User opens the application
          │
          ▼
Fill Candidate Profile
(name, role, experience, skills, resume summary)
          │
          ▼
Select Interview Configuration
(type, number of questions, difficulty)
          │
          ▼
Click "Generate My Interview Kit"
          │
          ▼
ProfileParser
          │
          ▼
UserProfile
          │
          ▼
InterviewOrchestrator
          │
          ├── Question Generator
          │       ├── RAG Retrieval
          │       └── IBM Granite
          ├── Answer Generator
          │       ├── RAG Retrieval
          │       └── IBM Granite
          ├── Feedback / Tips
          └── Preparation Strategy
          │
          ▼
Interview Kit
          ├── Overview
          ├── Questions
          ├── Model Answers
          └── Tips & Strategy
          │
          ▼
Optional Features
          ├── Evaluate Answer
          ├── Mock Interview
          └── Final Feedback

🔎 Retrieval-Augmented Generation (RAG)

The project uses a RAG architecture to retrieve relevant interview information before generating personalized content.

RAG Pipeline

Knowledge Base
      │
      ▼
Document Loader
      │
      ▼
Text Chunking
      │
      ▼
Sentence Transformer Embeddings
      │
      ▼
ChromaDB Vector Store
      │
      ▼
Semantic Retrieval
      │
      ▼
Relevant Context
      │
      ▼
IBM Granite
      │
      ▼
Personalized Interview Content

RAG helps the application provide context-aware interview questions and preparation content instead of relying only on the language model.

🤖 IBM Granite

The primary language model used by the application is:

ibm/granite-4-h-small

The model is accessed through IBM watsonx.ai / Watson Machine Learning.

WATSONX_API_KEY=your_ibm_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-4-h-small

🛡️ Local Fallback Mode

The application includes a local fallback mechanism to keep the interview trainer functional when IBM Granite is temporarily unavailable.

Fallback conditions can include:

IBM watsonx.ai token quota exhaustion

HTTP 403 or 429 responses

Rate-limit errors

Service availability issues

Missing IBM credentials

When IBM Granite is unavailable, the application switches to local deterministic generation/evaluation for supported interview flows.

⚡ Local Fallback Mode — IBM Granite was unavailable.

🛠️ Technology Stack

Component

Technology

Purpose

LLM

IBM Granite ibm/granite-4-h-small

AI-powered text generation

AI Platform

IBM watsonx.ai

IBM AI/LLM platform

Runtime

IBM Watson Machine Learning

Model/API runtime

Embeddings

sentence-transformers/all-MiniLM-L6-v2

Local text embeddings

Vector Store

ChromaDB

Persistent vector database

RAG Framework

LangChain

RAG pipeline and document processing

UI

Gradio

Interactive web interface

Language

Python 3.10+

Application development

Testing

Pytest

Automated testing

Configuration

python-dotenv

Environment configuration

📁 Project Structure

AI-Interview-Trainer-Agent/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── CONTRIBUTING.md
├── ai-interview-trainer-plan.md
├── agent/
├── config/
├── llm/
├── rag/
├── prompts/
├── utils/
├── knowledge_base/
├── chroma_db/
└── tests/

🚀 Setup & Installation

Prerequisites

Python 3.10 or higher

IBM Cloud account

IBM watsonx.ai project

IBM Cloud API Key

Git

Step 1 — Clone the Repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Interview-Trainer-Agent

Step 2 — Create a Virtual Environment

Windows

python -m venv .venv
.venv\Scripts\activate

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate

Step 3 — Install Dependencies

pip install -r requirements.txt

Step 4 — Configure Environment Variables

copy .env.example .env

Add your IBM credentials to .env:

WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-4-h-small
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
TOP_K_RETRIEVAL=5

⚠️ Never commit .env to GitHub.

Step 5 — Build the Knowledge Base

python -m rag.ingest

Step 6 — Launch the Application

python app.py

Open:

http://127.0.0.1:7860

🖥️ Application Tabs

Tab

Description

📋 Overview

Displays the generated interview kit summary

❓ Questions

Displays questions categorized by interview type

📝 Model Answers

Provides interview-ready model answers

💡 Tips & Strategy

Provides preparation tips and strategy

🔍 Evaluate Answer

Evaluates an individual candidate answer

🎤 Mock Interview

Conducts an interactive interview session

📊 Final Feedback

Provides overall performance feedback

📊 Interview Question Categories

The application supports four main categories:

Technical

Role-Specific

HR

Behavioral

Behavioral answers can use the STAR method:

Situation
Task
Action
Result

📝 Example Candidate Profile

Name: Candidate
Role: Software Engineer
Experience: Fresher (0-1 years)
Skills: Python, SQL, Machine Learning, RAG,
IBM Granite, ChromaDB, Git, REST API

📌 Example Interview Configuration

Interview Type: Mixed
Questions per Type: 5
Difficulty: Auto

The application generates:

5 Technical Questions
5 Role-Specific Questions
5 HR Questions
5 Behavioral Questions

Total = 20 Questions

🔍 Answer Evaluation

The evaluator provides:

Overall score

Strengths

Weaknesses

Missing points

Technical accuracy

Communication feedback

Improvement suggestions

Model answer

🎤 Mock Interview

Start Mock Interview
        ↓
Receive Question
        ↓
Enter Answer
        ↓
Submit Answer
        ↓
Receive Feedback
        ↓
Next Question
        ↓
Complete Interview
        ↓
View Average Score

📊 Final Feedback

After completing the mock interview, the system provides:

Number of questions answered

Average score

Overall assessment

Strengths

Areas for improvement

Personalized preparation recommendations

7-day preparation plan

🧪 Testing

Run the complete test suite:

pytest tests/ -v

Basic Python syntax check:

python -m py_compile app.py

🔐 Security

Never commit API credentials to GitHub.

The following should remain excluded through .gitignore:

.env
.venv/
__pycache__/
*.pyc
*.pyo

📸 Project Demonstration Screenshots

Recommended screenshots:

IBM watsonx.ai / Granite configuration

Project architecture / RAG workflow

Candidate profile and Interview Kit configuration

Generated Questions

Model Answers

Tips & Strategy

Answer Evaluation

Mock Interview / Final Feedback

🎓 AICTE IBM SkillsBuild Context

AICTE IBM SkillsBuild
Problem Statement No. 22
AI Interview Trainer Agent

The project demonstrates:

Generative AI

Large Language Models

IBM Granite

IBM watsonx.ai

Retrieval-Augmented Generation

Vector databases

Prompt engineering

AI-based answer evaluation

Interactive mock interviews

Personalized AI feedback

⚠️ Limitations

IBM watsonx.ai usage is subject to available service quotas.

When IBM Granite is unavailable, supported application flows use Local Fallback Mode.

The current knowledge base supports a defined set of job roles and interview content.

The application interface and knowledge base are primarily designed for English.

Sessions are not permanently stored between application restarts.

ChromaDB runs locally.

The application is intended primarily for educational and demonstration purposes.

🔮 Future Improvements

Add more job roles

Add PDF resume upload and automatic parsing

Add persistent interview history

Add speech-to-text input

Add voice-based mock interviews

Support multiple languages

Add advanced interview analytics

Add confidence scoring

Add visualization dashboards

Deploy the application to IBM Cloud

Add user authentication and session management

📦 Recommended GitHub Repository Information

Repository Name:

AI-Interview-Trainer-Agent

Description:

AI Interview Trainer Agent using RAG, IBM Granite, ChromaDB and Gradio for AICTE IBM SkillsBuild Problem Statement #22.

👤 Author

Sanket Kolhe

AICTE IBM SkillsBuild
Problem Statement No. 22

📄 License

This project was developed as part of the AICTE IBM SkillsBuild programme for educational and academic purposes.

🙏 Acknowledgements

AICTE & IBM SkillsBuild

IBM watsonx.ai

IBM Watson Machine Learning

LangChain

ChromaDB

Gradio

Hugging Face

⭐ Project Summary

The AI Interview Trainer Agent provides an end-to-end AI-powered interview preparation experience.

Candidate Profile
       ↓
RAG Retrieval
       ↓
IBM Granite
       ↓
Personalized Interview Kit
       ↓
Questions + Model Answers
       ↓
Answer Evaluation
       ↓
Mock Interview
       ↓
Final Feedback
       ↓
7-Day Preparation Strategy

AICTE IBM SkillsBuild — Problem Statement #22

AI Interview Trainer Agent