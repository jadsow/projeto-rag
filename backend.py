from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

# Ferramentas do LangChain para a lógica RAG
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# Importação corrigida para remover o aviso de depreciação
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CONFIGURAÇÃO INICIAL ---

print("Backend iniciando...")
print("Carregando o banco de dados vetorial e a cadeia RAG...")

# Definição das constantes
NOME_MODELO_EMBEDDING = "all-MiniLM-L6-v2"
PASTA_DB = "db"
NOME_MODELO_LLM_LOCAL = "llama3" # Para performance máxima em CPU, considere "phi3:mini"

modelo_embedding = HuggingFaceEmbeddings(model_name=NOME_MODELO_EMBEDDING)
db = Chroma(persist_directory=PASTA_DB, embedding_function=modelo_embedding)


host_ollama = os.getenv("OLLAMA_HOST", "localhost")
llm = ChatOllama(
    base_url=f"http://{host_ollama}:11434",
    model=NOME_MODELO_LLM_LOCAL,
    num_predict=150  
)

retriever = db.as_retriever(search_kwargs={'k': 3})


template = """
<instrucoes>
Você é um assistente de IA especialista em análise de documentos. Sua missão é responder à pergunta do usuário de forma útil, precisa e CONCISA, baseando-se estritamente no conteúdo encontrado no bloco <contexto>.

## REGRAS DE OURO ##
- **Seja Direto:** NUNCA mencione o contexto, o documento ou de onde você tirou a informação. Apenas forneça a resposta. Não use frases como "Baseado no contexto...", "O documento afirma que...".
- **Seja Conciso:** Mantenha suas respostas o mais breve possível, idealmente entre 1 a 3 frases.
- **Nunca invente informações.** Se não houver resposta no contexto, siga a hierarquia definida abaixo.
- **Nunca quebre o tom da interação.** Se o usuário apenas cumprimentar, responda educadamente apenas o cumprimento, sem adicionar nada a mais.
- **Nunca retorne tags (<contexto>, <pergunta>, etc.) na resposta. Apenas o conteúdo.**

## HIERARQUIA DE RESPOSTA ##
1. Se o usuário apenas fizer uma saudação (ex: "Oi", "Bom dia", "Olá"), responda apenas com a mesma saudação.
2. Se o usuário fizer uma saudação + pergunta ou pedido, retribua a saudação e em seguida responda ao pedido com base no contexto.
   - Exemplo: Usuário: "Bom dia, qual o horário de funcionamento?"  
     Resposta: "Bom dia! O horário de funcionamento é das 9h às 18h."
3. Se o usuário apenas fizer uma pergunta, responda diretamente sem adicionar saudação.
4. Verifique se o contexto contém uma resposta direta e explícita para a pergunta. Se sim, forneça essa resposta, seguindo as Regras de Ouro.
5. Se não houver uma resposta direta, procure pela informação mais relevante no contexto e comece sua resposta EXATAMENTE com a frase:  
   "Não encontrei uma resposta direta, mas aqui está uma informação relacionada:"
6. Se o contexto não tiver nenhuma informação relevante, responda EXATAMENTE com a frase:  
   "Não encontrei informações sobre este tópico nos documentos."
7. Se o usuário pedir algo fora do escopo (piadas, opiniões pessoais, assuntos não relacionados), responda EXATAMENTE com a frase:  
   "Posso responder apenas perguntas relacionadas ao conteúdo dos documentos."
   
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

        resultado = cadeia_rag.invoke({"input": pergunta.texto})

        print("--- Contexto usado para gerar a resposta: ---")
        for i, doc in enumerate(resultado.get("context", [])):
             print(f"Trecho {i+1}:")
             print(doc.page_content)
             print("-" * 20)
        
        return {"resposta": resultado["answer"]}
    except Exception as e:
        return {"erro": f"Ocorreu um erro durante o processamento: {str(e)}"}