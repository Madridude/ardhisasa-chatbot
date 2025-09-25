import json
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def load_kb(json_file, collection_name, persist_dir):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [entry["question"] + " " + entry["answer"] for entry in data]
    metadatas = [{"section": entry["section"], "intent": entry["intent"]} for entry in data]

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    db.persist()

if __name__ == "__main__":
    load_kb("faq_vector_knowledge_base.json", "faq_kb", "chroma_faq")
    load_kb("sop_vector_knowledge_base.json", "sop_kb", "chroma_sop")
    print("Knowledge bases ingested successfully!")