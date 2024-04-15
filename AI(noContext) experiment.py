from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="./googleflanT5Large/instructorX1/instructor-xl") #this is local
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("\nVectorStore:",vectorstore, "\n")
    return vectorstore

def get_conversation_chain(vectorstore):
    # Load the local model components
    model_name = "./googleflanT5Large/flan-t5-large"
    llm = HuggingFacePipeline.from_model_id(model_id= model_name, task="text2text-generation", model_kwargs={"temperature": 0, "max_length": 200})
    conversation_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    return conversation_chain


def handle_userinput(conversation_chain):
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Thank you for using us!")
            break
        result = conversation_chain.invoke({"query": user_input})

        response = result["result"]

        print("Chatbot:", response)


def main():
    load_dotenv()

    conversation = {'question': None, 'answer': None}
    chat_history = None

    pdf_docs = ["pdf2.pdf"]
    print("Your documents: ", pdf_docs)
    raw_text = get_pdf_text(pdf_docs)
    print("Text extracted from your documents")
    text_chunks = get_text_chunks(raw_text)
    print("done getting text chunks")
    vectorstore = get_vectorstore(text_chunks)
    print("done making vectorstore")
    
    conversation_chain = get_conversation_chain(vectorstore)
    handle_userinput(conversation_chain)

if __name__ == '__main__':
    main()