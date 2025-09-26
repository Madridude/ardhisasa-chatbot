from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    faq_db = Chroma(persist_directory="chroma_faq", collection_name="faq_kb", embedding_function=None)
    sop_db = Chroma(persist_directory="chroma_sop", collection_name="sop_kb", embedding_function=None)

    faq_qa = RetrievalQA.from_chain_type(llm=llm, retriever=faq_db.as_retriever())
    sop_qa = RetrievalQA.from_chain_type(llm=llm, retriever=sop_db.as_retriever())

    print("Ardhisasa Dual Chatbot (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        if any(word in query.lower() for word in ["password", "login", "account", "transfer", "payment", "parcel"]):
            result = faq_qa.run(query)
            print("FAQ KB:", result)
        else:
            result = sop_qa.run(query)
            print("SOP KB:", result)

if __name__ == "__main__":
    main()
