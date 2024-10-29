# app_streamlit.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_ollama import OllamaLLM
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Configuración de la página
st.set_page_config(
    page_title="Asistente REDOAPE",
    page_icon="🤖",
    layout="centered"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.assistant {
        background-color: #f0f0f0;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        margin-right: 1rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Inicialización de la sesión
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu asistente virtual especializado en documentos REDOAPE del Ejército Argentino. ¿En qué puedo ayudarte?"}
    ]

if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None

def load_qa_system():
    """Carga o inicializa el sistema de QA"""
    if st.session_state.qa_system is None:
        with st.spinner('Cargando el sistema...'):
            # Cargar el índice
            embeddings = HuggingFaceInstructEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            document_search = FAISS.load_local(
                "faiss_indexes/archivos_procesados",
                embeddings,
                allow_dangerous_deserialization=True
            )

            # Configurar el LLM y el prompt
            llm = OllamaLLM(model="llama3",temperature=0.1)
            prompt_template = """Eres un asistente virtual especializado en los documentos REDOAPE relacionados con el Ejército Argentino. 
            Tu conocimiento abarca reglamentos, procedimientos, jerarquías y normativas específicas de la institución.

            Instrucciones:
            1. Responde siempre en español
            2.  No inventes ni asumas información adicional.
            3. Si la información en el contexto es insuficiente para responder completamente, indica claramente qué información adicional sería necesaria.
            4. Si encuentras información contradictoria en el contexto, señálalo en tu respuesta y explica las discrepancias.
            5. Organiza tu respuesta de manera clara y estructurada, utilizando viñetas o numeración si es necesario para mejorar la legibilidad.

            Contexto proporcionado:
            {context}

            Pregunta del usuario: {question}

            Respuesta detallada y concisa:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Crear el sistema QA
            st.session_state.qa_system = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=document_search.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

def display_messages():
    """Muestra los mensajes del chat"""
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user">
                        <div class="avatar">👤</div>
                        <div class="message">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message assistant">
                        <div class="avatar">🤖</div>
                        <div class="message">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

def main():
    st.title("💬 Asistente REDOAPE")
    st.markdown("---")

    # Cargar el sistema QA
    load_qa_system()

    # Mostrar mensajes
    display_messages()

    # Input del usuario
    if question := st.chat_input("Escribe tu pregunta aquí..."):
        # Agregar pregunta del usuario
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Obtener respuesta
        with st.spinner('Procesando tu pregunta...'):
            try:
                result = st.session_state.qa_system({"query": question})
                response = result["result"]
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = "Lo siento, hubo un error al procesar tu pregunta. Por favor, intenta de nuevo."
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.error(f"Error: {str(e)}")
        
        # Recargar mensajes
        st.rerun()

if __name__ == "__main__":
    main()