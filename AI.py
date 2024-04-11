from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

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
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = HuggingFaceInstructEmbeddings(model_name="./googleflanT5Large/instructorX1/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.5, "max_length":512})
    # llm = HuggingFaceHub(repo_id="lmsys/fastchat-t5-3b-v1.0", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(user_question, conversation):
    response = conversation({'question': user_question})
    chat_history = response['chat_history']
    print("Chat history:", chat_history)
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            print(f"User: {message.content}")
        else:
            print(f"Bot: {message.content}")

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

    while True:
        user_question = input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question, conversation_chain)

if __name__ == '__main__':
    main()