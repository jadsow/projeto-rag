from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_chroma.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings

PASTA_BASE = "base"

def criar_db():
  documentosbd = carregar_documentos()

  chunks = dividir_chunks(documentosbd)
  vetorizar_chunks(chunks)

def carregar_documentos():
  carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
  documentos = carregador.load()
  return documentos

def dividir_chunks(documentosbd):
  separador_de_documentos = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    add_start_index=True
  )

  chunks = separador_de_documentos.split_documents(documentosbd)
  print(len(chunks))
  return chunks

def vetorizar_chunks(chunks):
  modelo_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  db = Chroma.from_documents(chunks, modelo_embedding, persist_directory="db")

  print("Banco de dados criado")


criar_db()