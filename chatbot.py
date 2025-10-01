from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from openai.error import RateLimitError

def main():
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        faq_db = Chroma(
            persist_directory="chroma_faq",
            collection_name="faq_kb",
            embedding_function=None
        )
        sop_db = Chroma(
            persist_directory="chroma_sop",
            collection_name="sop_kb",
            embedding_function=None
        )

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
            except RateLimitError:
                print("⚠️ API quota exceeded. Please check your OpenAI billing plan.")

    except RateLimitError:
        print("⚠️ Could not start chatbot: API quota exceeded. Please check your OpenAI billing plan.")

if __name__ == "__main__":
    main()
