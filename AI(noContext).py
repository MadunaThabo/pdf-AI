from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import HuggingFaceHub
from transformers import AutoModelForSeq2SeqLM
from langchain.llms import google_palm as LLM





import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import ConversationalPipeline


from transformers import Conversation
from transformers import ConversationalPipeline


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
    embeddings = HuggingFaceInstructEmbeddings(model_name="./googleflanT5Large/instructorX1/instructor-xl") #this is local
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("\nVectorStore:",vectorstore, "\n")
    return vectorstore

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    model_name = "./googleflanT5Large/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.temperature = 0.5  # Change the temperature here
    conversation_chain = ConversationalPipeline(model=model,tokenizer=tokenizer)
    return conversation_chain


def handle_userinput(conversation_chain):
    conversation_history = []
    while True:
        user_input = input("You: ")
        conversation = Conversation(user_input)
        response = conversation_chain([conversation])
        print("Bot:", response)


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