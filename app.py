import os
import json
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFaceHub, Ollama
from openai import RateLimitError

st.set_page_config(page_title="Ardhisasa Dual Chatbot", layout="wide")

st.title("🤖 Ardhisasa Dual Chatbot")
st.write("Handles both **citizen FAQs** and **internal SOP queries** automatically with hybrid LLM fallback.")

# -------------------------------
# Sidebar: Choose LLM Provider
# -------------------------------
provider_choice = st.sidebar.radio(
    "Select LLM Provider",
    ["Auto (fallback)", "OpenAI", "Hugging Face", "Ollama", "Dummy"]
)

def get_llm(choice):
    """Return chosen LLM or auto-fallback."""
    if choice == "OpenAI":
        try:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception as e:
            st.error(f"⚠️ OpenAI error: {e}")
            return None

    if choice == "Hugging Face":
        try:
            return HuggingFaceHub(
                repo_id="google/flan-t5-base",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )
        except Exception as e:
            st.error(f"⚠️ Hugging Face error: {e}")
            return None

    if choice == "Ollama":
        try:
            return Ollama(model="llama2")
        except Exception as e:
            st.error(f"⚠️ Ollama error: {e}")
            return None

    if choice == "Dummy":
        class DummyLLM:
            def predict(self, text):
                return f"[Simulated Answer] You asked: {text}"
        return DummyLLM()

    # Auto fallback mode
    try:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except RateLimitError:
        st.warning("⚠️ OpenAI quota exceeded, switching to Hugging Face...")
    except Exception:
        pass

    try:
        return HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
    except Exception:
        pass

    try:
        return Ollama(model="llama2")
    except Exception:
        pass

    class DummyLLM:
        def predict(self, text):
            return f"[Simulated Answer] You asked: {text}"
    return DummyLLM()

# -------------------------------
# Step 1: Ensure vector DBs exist
# -------------------------------
embeddings = OpenAIEmbeddings()

def ingest_if_missing():
    if not os.path.exists("chroma_faq") or not os.path.exists("chroma_sop"):
        st.info("⚡ Running ingestion for the first time... please wait.")

        def load_kb(json_file, collection_name, persist_dir):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts = [entry["question"] + " " + entry["answer"] for entry in data]
            metadatas = [{"section": entry["section"], "intent": entry["intent"]} for entry in data]

            db = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                collection_name=collection_name,
                persist_directory=persist_dir,
            )
            db.persist()

        load_kb("faq_vector_knowledge_base.json", "faq_kb", "chroma_faq")
        load_kb("sop_vector_knowledge_base.json", "sop_kb", "chroma_sop")
        st.success("✅ Ingestion completed successfully!")

ingest_if_missing()

# -------------------------------
# Step 2: Load databases
# -------------------------------
faq_db = Chroma(persist_directory="chroma_faq", collection_name="faq_kb", embedding_function=embeddings)
sop_db = Chroma(persist_directory="chroma_sop", collection_name="sop_kb", embedding_function=embeddings)

llm = get_llm(provider_choice)
faq_qa = RetrievalQA.from_chain_type(llm=llm, retriever=faq_db.as_retriever())
sop_qa = RetrievalQA.from_chain_type(llm=llm, retriever=sop_db.as_retriever())

# -------------------------------
# Step 3: Chat interface
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything about Ardhisasa...")

if query:
    try:
        if any(word in query.lower() for word in ["password", "login", "account", "transfer", "payment", "parcel"]):
            answer = faq_qa.run(query)
            source = "FAQ KB"
        else:
            answer = sop_qa.run(query)
            source = "SOP KB"
    except Exception as e:
        answer = f"⚠️ Chatbot error: {e}"
        source = "System"

    st.session_state.history.append(("You", query))
    st.session_state.history.append((f"{source}", answer))

for speaker, message in st.session_state.history:
    st.markdown(f"**{speaker}:** {message}")
