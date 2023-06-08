import os
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from rich import print
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = "your openAI API key"

DATA_PATH = r'C:\Users\ACER\Downloads\OpenAI ChatBot\docs' #data path
MODEL_NAME = 'gpt-3.5-turbo' #gpt-4
CROMADB_DIRECTORY = 'db/'
RETURN_SOURCE_DOCUMENT = True

llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME, max_tokens=1000)

txtLoader = DirectoryLoader(DATA_PATH, glob="**/*.txt", show_progress=True)
pdfLoader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", show_progress=True)
mdLoader = DirectoryLoader(DATA_PATH, glob="**/*.md", show_progress=True)
csvLoader = DirectoryLoader(DATA_PATH, glob="**/*.csv", show_progress=True)
loaders = [txtLoader, pdfLoader, mdLoader, csvLoader]

documents = []
for loader in loaders:
    documents.extend(loader.load())

print(f"Total number of document: {len(documents)}")

textSplitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
documents = textSplitter.split_documents(documents)
print(f"Total number of document after split: {len(documents)}")

print("Creating Vector Store...")
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
vectorStore = Chroma.from_documents(documents, embeddings, persist_directory=CROMADB_DIRECTORY)
vectorStore.persist()

# metode QA yang bisa digunakan:
# - Similarity Search
# - RetrievalQA
# - RetrievalQA with Prompt
# - ConversationalRetrievalChain



"""template = 
jawablah pertanyaan {question} berdasarkan konteks yang ada, dan katakan "tidak ada pada konteks" jika jawaban tidak ada pada konteks 

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)"""

qa = ConversationalRetrievalChain.from_llm(llm, vectorStore.as_retriever(), return_source_documents=RETURN_SOURCE_DOCUMENT)

# Your Question and Answer

chat_history = []
#question = "jelaskan Undang-Undang Nomor 4 Tahun 1984 ? tolong jelaskan dalam 2 paragraf"

def responseQuery(question):
    response = qa({"question": question, "chat_history": chat_history, "answer" : []})
    chat_history_user = {"role" : "user","content" : question}
    chat_history_system = {"role" : "system","content" : response["answer"]}
    chat_history.append((chat_history_user, chat_history_system))
    return response


#print(response)
#print()
#print(response["prompt"])
#print(response["answer"])


## delete chroma DB
#vectorStore.delete_collection()
#vectorStore.persist()
