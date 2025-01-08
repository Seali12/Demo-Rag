import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
import os

st.set_page_config(page_title="Asistente REDOAPE", page_icon="🤖", layout="wide")

# Función para obtener contexto
def obtener_contexto(retriever, pregunta, k=5):
    try:
        resultados = retriever.get_relevant_documents(pregunta)
        contexto = "\n".join([doc.page_content for doc in resultados])
        return contexto
    except Exception as e:
        st.error(f"Error al obtener contexto: {e}")
        return ""

def mejorar_pregunta(pregunta, contexto):
    llm = OllamaLLM(model="llama3", temperature=0.1)
    prompt = f"""
    Eres un asistente especializado en reformular preguntas en ESPAÑOL, 
    mejorando su claridad, precisión y redacción. IMPORTANTE: Todas las 
    respuestas deben ser COMPLETAMENTE EN ESPAÑOL.
    
    Contexto disponible: {contexto}
    
    Pregunta original: {pregunta}
    
    Tu tarea es generar 3 versiones diferentes de la pregunta original, 
    SOLO EN ESPAÑOL:
    1. Mantén el significado original, mejorando la redacción
    2. Reformula la pregunta de manera más específica y precisa
    3. Ajusta el lenguaje para que sea más formal o clara
    
    Reglas estrictas:
    - TODAS las preguntas DEBEN estar en ESPAÑOL
    - No uses ninguna palabra en otro idioma
    - Mantén el sentido original de la pregunta
    - Asegúrate de que la pregunta sea gramaticalmente correcta
    
    Devuelve SOLO las 3 preguntas reformuladas, una por línea, 
    sin numeración ni explicaciones adicionales.
    """
    
    try:
        result = llm(prompt)
        # Limpiar y filtrar las opciones
        opciones = [
            opcion.strip() 
            for opcion in result.split("\n") 
            if opcion.strip() and 
               len(opcion.strip()) > 10 and 
               # Filtro adicional para asegurar que sea español
               not any(palabra in opcion.lower() for palabra in ['what', 'how', 'why', 'where', 'when'])
        ]
        
        if len(opciones) < 3:
            opciones = [pregunta]
        
        opciones.insert(0, pregunta)
        
        return opciones[:4]  # Limitar a 4 opciones
    
    except Exception as e:
        st.error(f"Error al generar preguntas mejoradas: {e}")
        return [pregunta]  # Devolver la pregunta original en caso de error

# Cargar el sistema QA
def load_qa_system():
    if 'qa_system' not in st.session_state:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                model_kwargs={"device": "cpu"}
            )
            index = FAISS.load_local(
                "faiss_indexes/archivos_procesados", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            retriever = index.as_retriever(search_kwargs={"k": 15})
            
            llm = OllamaLLM(model="llama3", temperature=0.1)
            prompt_template = """Eres un asistente especializado en documentos REDOAPE. 
            Contexto: {context}
            Pregunta: {question}
            Responde de manera clara, concisa y profesional:
            """
            
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

# Mostrar el historial de chat
def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

# Función principal de Streamlit
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