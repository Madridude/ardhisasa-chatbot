import os
import json
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from openai.error import RateLimitError

st.set_page_config(page_title="Ardhisasa Dual Chatbot", layout="wide")

st.title("🤖 Ardhisasa Dual Chatbot")
st.write("Handles both **citizen FAQs** and **internal SOP queries** automatically.")

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
faq_db = Chroma(
    persist_directory="chroma_faq",
    collection_name="faq_kb",
    embedding_function=embeddings
)
sop_db = Chroma(
    persist_directory="chroma_sop",
    collection_name="sop_kb",
    embedding_function=embeddings
)

try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    faq_qa = RetrievalQA.from_chain_type(llm=llm, retriever=faq_db.as_retriever())
    sop_qa = RetrievalQA.from_chain_type(llm=llm, retriever=sop_db.as_retriever())
except RateLimitError:
    st.error("⚠️ Your OpenAI quota has been exceeded. Please check your billing plan.")
    st.stop()

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
    except RateLimitError:
        answer = "⚠️ API quota exceeded. Please upgrade your OpenAI plan."
        source = "System"

    st.session_state.history.append(("You", query))
    st.session_state.history.append((f"{source}", answer))

for speaker, message in st.session_state.history:
    st.markdown(f"**{speaker}:** {message}")
