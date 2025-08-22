# frontend.py

import streamlit as st
import requests
import os

st.set_page_config(page_title="Chat com Documentos", page_icon="📄")
st.title("📄 Chatbot de Gestão de Documentos")
st.caption("Este é um chatbot que tem por objetivo tirar dúvidas que envolvem o módulo de Gestão de Documentos do NIFE")

host_backend = os.getenv("BACKEND_HOST", "localhost")
API_URL = f"http://{host_backend}:8000/perguntar"


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Olá! Qual sua dúvida referente ao módulo de Gestão de Documentos?"
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Pensando...")

        try:
            payload = {"texto": prompt}
            
            response = requests.post(API_URL, json=payload)
            response.raise_for_status() 

            full_response = response.json().get("resposta", "Desculpe, não consegui encontrar uma resposta.")
            
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            error_message = f"Não foi possível conectar ao backend. Verifique se ele está rodando. (Erro: {e})"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})