from fastapi import FastAPI
from pydantic import BaseModel

# Ferramentas do LangChain para a lógica RAG
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama 
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from fastapi.middleware.cors import CORSMiddleware

import os


print("Backend iniciando...")
print("Carregando o banco de dados vetorial e a cadeia RAG...")

NOME_MODELO_EMBEDDING = "all-MiniLM-L6-v2"
PASTA_DB = "db"
NOME_MODELO_LLM_LOCAL = "llama3"

modelo_embedding = HuggingFaceEmbeddings(model_name=NOME_MODELO_EMBEDDING)

db = Chroma(persist_directory=PASTA_DB, embedding_function=modelo_embedding)

vector_db_retriever = db.as_retriever(search_kwargs={'k': 3})

host_ollama = os.getenv("OLLAMA_HOST", "localhost")
llm = ChatOllama(base_url=f"http://{host_ollama}:11434", model=NOME_MODELO_LLM_LOCAL)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_db_retriever, llm=llm
)


template = """
<instrucoes>
Você é um assistente de IA especialista em análise de documentos. Sua missão é responder à pergunta do usuário de forma útil, precisa e CONCISA, baseando-se estritamente no conteúdo encontrado no bloco <contexto>.

## REGRAS DE OURO ##
- **Lógica Negativa:** Se a resposta factual no contexto for uma negação (ex: "Não é possível", "Não permite"), sua resposta final DEVE começar diretamente com "Não". Jamais comece uma resposta negativa com "Sim".
- **Seja Direto:** NUNCA mencione o contexto, o documento ou de onde você tirou a informação. Apenas forneça a resposta.
- **Seja Conciso:** Mantenha suas respostas o mais breve possível, idealmente entre 1 a 3 frases.

## HIERARQUIA DE RESPOSTA ##
1.  Verifique se o contexto contém uma resposta direta e explícita para a pergunta. Se sim, forneça essa resposta, seguindo as Regras de Ouro.
2.  Se não houver uma resposta direta, procure pela informação mais relevante no contexto. Ao apresentar essa informação, **comece sua resposta EXATAMENTE com a frase:** "Não encontrei uma resposta direta, mas aqui está uma informação relacionada:"
3.  Se o contexto não tiver nenhuma informação relevante, responda EXATAMENTE com a frase: "Não encontrei informações sobre este tópico nos documentos."

Suas instruções terminam aqui.
</instrucoes>

<contexto>
{context}
</contexto>

<pergunta>
{input}
</pergunta>

<resposta>
"""

prompt = ChatPromptTemplate.from_template(template)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)

cadeia_rag = create_retrieval_chain(retriever, combine_docs_chain)

print("Backend pronto e aguardando requisições.")


app = FastAPI(
    title="API do Chatbot RAG",
    description="Uma API para conversar com documentos usando RAG e Llama3.",
    version="1.0.0"
)

origins = [
    "http://localhost:4200",  
    "http://localhost",     
    "*"                      
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      
    allow_credentials=True,
    allow_methods=["*"],        
    allow_headers=["*"],        
)

class Pergunta(BaseModel):
    texto: str

@app.post("/perguntar")
def perguntar_ao_rag(pergunta: Pergunta):
    """
    Recebe uma pergunta em texto e retorna a resposta gerada pelo modelo RAG.
    """
    try:
        print(f"--- Recebida a pergunta: {pergunta.texto} ---")
        
        trechos_retornados = retriever.invoke(pergunta.texto)
        
        print("--- Contexto que será enviado ao LLM: ---")
        for i, doc in enumerate(trechos_retornados):
            print(f"Trecho {i+1}:")
            print(doc.page_content) 
            print("-" * 20)


        resultado = cadeia_rag.invoke({"input": pergunta.texto})
        
        return {"resposta": resultado["answer"]}
    except Exception as e:
        return {"erro": f"Ocorreu um erro durante o processamento: {str(e)}"}