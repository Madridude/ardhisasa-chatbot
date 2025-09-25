# Ardhisasa Dual Chatbot (FAQ + SOP)

This project provides a dual-knowledge AI chatbot that answers both **citizen-facing FAQs** and **internal staff SOP rules**.

## Project Structure
- ingest.py – Load KBs into Chroma
- chatbot.py – Terminal chatbot
- app.py – Streamlit browser chatbot
- faq_vector_knowledge_base.json – Citizen KB (with samples)
- sop_vector_knowledge_base.json – Staff KB (with samples)
- requirements.txt – Dependencies
- Dockerfile – For container deployment
- README.md – Quick start guide

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Set your API key: `export OPENAI_API_KEY="your-key"`
3. Ingest KBs: `python ingest.py`
4. Run terminal chatbot: `python chatbot.py`
5. Run Streamlit app: `streamlit run app.py`

## Deployment
- Streamlit Cloud (recommended)
- Hugging Face Spaces
- Docker (Heroku, Render, VPS)
