import streamlit as st
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

st.set_page_config(page_title="Asistente REDOAPE", page_icon="🤖", layout="centered")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Corriendo en: {device}")

def obtener_contexto(retriever, pregunta, k=5):
    try:
        resultados = retriever.get_relevant_documents(pregunta)
        contexto = "\n".join([doc.page_content for doc in resultados])
        return contexto
    except Exception as e:
        st.error(f"Error al obtener contexto: {e}")
        return ""

def mejorar_pregunta(pregunta, contexto):
    llm = OllamaLLM(
        model="deepseek-r1:14b",
        temperature=0,
        extra_model_kwargs={
            "gpu": True,
            "n_gpu_layers": 35
        } if device == "cuda" else {}
    )
    prompt = f"""Como asistente en español, tu tarea es reformular la siguiente pregunta de 3 formas diferentes.

Reglas:
1. SOLO proporciona las preguntas reformuladas, nada más
2. Cada pregunta debe estar en una línea separada
3. NO incluyas números, explicaciones ni etiquetas
4. Las preguntas deben ser más específicas que la original
5. Usa SOLO español
6. NO uses etiquetas como <think> o similares

Pregunta original: {pregunta}

Reformula la pregunta de 3 formas diferentes:"""

    try:
        result = llm(prompt)
        opciones = [opcion.strip() for opcion in result.split("\n") if opcion.strip() and not opcion.startswith("<")]
        if len(opciones) < 3:
            opciones = [pregunta]
        opciones.insert(0, pregunta)
        return opciones[:4]
    except Exception as e:
        st.error(f"Error al generar preguntas mejoradas: {e}")
        return [pregunta]

def load_qa_system():
    if 'qa_system' not in st.session_state:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                model_kwargs={"device": device}
            )
            
            index = FAISS.load_local(
                "faiss_indexes/archivos_procesados",
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = index.as_retriever(search_kwargs={"k": 10})
            
            llm = OllamaLLM(
                model="deepseek-r1:14b",
                temperature=0,
                extra_model_kwargs={
                    "gpu": True,
                    "n_gpu_layers": 35
                } if device == "cuda" else {}
            )
            
            prompt_template = """Eres un asistente experto que responde preguntas sobre documentos REDOAPE.

REGLAS ESTRICTAS:
1. Responde SOLO con la información encontrada en el contexto
2. Da respuestas CONCISAS y DIRECTAS
3. NO uses más de 3 oraciones en tu respuesta
4. NO des explicaciones adicionales
5. NO uses etiquetas como <think>
6. Si no hay información en el contexto, responde SOLO: "No encontré información sobre esto en los documentos."
7. SOLO español

Contexto: {context}

Pregunta: {question}

Respuesta concisa:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            st.session_state.qa_system = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
            )
            st.session_state.retriever = retriever
        except Exception as e:
            st.error(f"Error al cargar el sistema QA: {e}")
            st.stop()

def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

def main():
    st.title("🤖 Asistente REDOAPE")
    st.markdown("Sistema de consultas inteligente para documentos.")

    load_qa_system()

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'preguntas_mejoradas' not in st.session_state:
        st.session_state.preguntas_mejoradas = None
    if 'show_improved_questions' not in st.session_state:
        st.session_state.show_improved_questions = False

    display_chat()

    user_input = st.chat_input("Escribe tu pregunta aquí...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        contexto = obtener_contexto(
            st.session_state.retriever,
            user_input
        )

        st.session_state.preguntas_mejoradas = mejorar_pregunta(
            user_input,
            contexto
        )

        st.session_state.show_improved_questions = True
        st.rerun()

    if st.session_state.show_improved_questions and st.session_state.preguntas_mejoradas:
        st.markdown("### 🔍 Opciones de preguntas mejoradas:")
        selected_question = st.radio(
            "Selecciona una pregunta para obtener una respuesta más precisa:",
            st.session_state.preguntas_mejoradas
        )

        if st.button("Obtener respuesta"):
            with st.spinner("Buscando la mejor respuesta..."):
                result = st.session_state.qa_system({"query": selected_question})
                response = result["result"]

                st.session_state.messages.append(
                    {"role": "user", "content": selected_question}
                )

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                st.session_state.show_improved_questions = False
                st.session_state.preguntas_mejoradas = None
            
            st.rerun()

if __name__ == "__main__":
    main()