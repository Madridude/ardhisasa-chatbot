from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFaceHub, Ollama
from openai.error import RateLimitError
import os

def get_llm():
    try:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except RateLimitError:
        print("⚠️ OpenAI quota exceeded, switching to Hugging Face...")
    except Exception as e:
        print(f"⚠️ OpenAI not available: {e}")

    try:
        return HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
    except Exception as e:
        print(f"⚠️ Hugging Face not available: {e}")

    try:
        return Ollama(model="llama2")
    except Exception as e:
        print(f"⚠️ Ollama not available: {e}")

    class DummyLLM:
        def predict(self, text):
            return f"[Simulated Answer] You asked: {text}"
    return DummyLLM()

def main():
    llm = get_llm()

    faq_db = Chroma(persist_directory="chroma_faq", collection_name="faq_kb", embedding_function=None)
    sop_db = Chroma(persist_directory="chroma_sop", collection_name="sop_kb", embedding_function=None)

    faq_qa = RetrievalQA.from_chain_type(llm=llm, retriever=faq_db.as_retriever())
    sop_qa = RetrievalQA.from_chain_type(llm=llm, retriever=sop_db.as_retriever())

    print("Ardhisasa Dual Chatbot (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        try:
            if any(word in query.lower() for word in ["password", "login", "account", "transfer", "payment", "parcel"]):
                result = faq_qa.run(query)
                print("FAQ KB:", result)
            else:
                result = sop_qa.run(query)
                print("SOP KB:", result)
        except Exception as e:
            print(f"⚠️ Chatbot error: {e}")

if __name__ == "__main__":
    main()
