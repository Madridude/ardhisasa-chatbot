import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma

st.set_page_config(page_title="Ardhisasa Dual Chatbot", layout="wide")

st.title("🤖 Ardhisasa Dual Chatbot")
st.write("Handles both **citizen FAQs** and **internal SOP queries** automatically.")

# Load databases
faq_db = Chroma(persist_directory="chroma_faq", collection_name="faq_kb", embedding_function=None)
sop_db = Chroma(persist_directory="chroma_sop", collection_name="sop_kb", embedding_function=None)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
faq_qa = RetrievalQA.from_chain_type(llm=llm, retriever=faq_db.as_retriever())
sop_qa = RetrievalQA.from_chain_type(llm=llm, retriever=sop_db.as_retriever())

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything about Ardhisasa...")

if query:
    if any(word in query.lower() for word in ["password", "login", "account", "transfer", "payment", "parcel"]):
        answer = faq_qa.run(query)
        source = "FAQ KB"
    else:
        answer = sop_qa.run(query)
        source = "SOP KB"

    st.session_state.history.append(("You", query))
    st.session_state.history.append((f"{source}", answer))

for speaker, message in st.session_state.history:
    st.markdown(f"**{speaker}:** {message}")