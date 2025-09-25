import os
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def main():
    print("✅ LangChain & OpenAI modules imported successfully!")

    # Test embeddings
    embeddings = OpenAIEmbeddings()
    test_vector = embeddings.embed_query("Hello Ardhisasa")
    print(f"✅ Embeddings working, sample vector length: {len(test_vector)}")

    # Test ChatOpenAI (requires OPENAI_API_KEY)
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.predict("Say 'Ardhisasa chatbot is ready!'")
        print("✅ ChatOpenAI working:", response)
    except Exception as e:
        print("⚠️ ChatOpenAI test skipped (no API key?):", e)

    # Test Chroma ingestion (expects chroma_faq/ and chroma_sop/ after running ingest.py)
    try:
        if os.path.exists("chroma_faq") and os.path.exists("chroma_sop"):
            faq_db = Chroma(persist_directory="chroma_faq", collection_name="faq_kb", embedding_function=embeddings)
            sop_db = Chroma(persist_directory="chroma_sop", collection_name="sop_kb", embedding_function=embeddings)
            print(f"✅ Chroma FAQ DB loaded with {faq_db._collection.count()} entries")
            print(f"✅ Chroma SOP DB loaded with {sop_db._collection.count()} entries")
        else:
            print("⚠️ Chroma DBs not found. Run `python ingest.py` first.")
    except Exception as e:
        print("❌ Error testing Chroma ingestion:", e)

if __name__ == "__main__":
    main()
