import streamlit as st
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import os

st.set_page_config(page_title="Asistente REDOAPE", page_icon="🤖", layout="centered")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Corriendo en: {device}")

def process_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            texto_crudo = ''
            for line in file:
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        texto_crudo += line.encode(encoding).decode('utf-8')
                        break
                    except UnicodeDecodeError:
                        continue
        return texto_crudo
    except Exception as e:
        st.error(f"Error procesando {txt_path}: {str(e)}")
        return ""

def process_folder(folder_path):
    all_text = ""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                st.info(f"Procesando: {file_path}")
                file_content = process_txt(file_path)
                if file_content:
                    all_text += file_content + "\n\n"

    if not all_text:
        raise ValueError("No se encontró contenido en la carpeta especificada.")
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=700,
        chunk_overlap=300,
        length_function=len,
    )

    return text_splitter.split_text(all_text)

def create_and_save_index(texts, index_name):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    
    document_search = FAISS.from_texts(texts, embeddings)
    
    os.makedirs("faiss_indexes", exist_ok=True)
    
    document_search.save_local(f"faiss_indexes/{index_name}")
    st.success(f"Índice guardado como {index_name}")

def initialize_faiss_index():
    folder_path = 'ArchivosParteEnfermo'
    index_name = 'archivos_procesados'
    index_path = f"faiss_indexes/{index_name}"

    if not os.path.exists(index_path):
        with st.spinner("Inicializando el sistema por primera vez..."):
            st.info("Procesando archivos...")
            texts = process_folder(folder_path)
            st.info("Creando índice FAISS...")
            create_and_save_index(texts, index_name)
    else:
        st.success("Índice FAISS encontrado. Sistema listo para usar.")

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
                model="llama3",
                temperature=0.2,
                extra_model_kwargs={
                    "gpu": True,
                    "n_gpu_layers": 35
                } if device == "cuda" else {}
            )
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
        opciones = [opcion.strip() for opcion in result.split("\n") if opcion.strip()]
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
            
            llm = OllamaLLM(model="llama3", temperature=0, 
                extra_model_kwargs={
                    "gpu": True,
                    "n_gpu_layers": 35
                } if device == "cuda" else {}
            )
            
            prompt_template = """Eres un asistente especializado en documentos con informacion sobre el Ejercito Argentino. 
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
            
def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

def main():
    st.title("🤖 Asistente REDOAPE")
    st.markdown("Sistema de consultas inteligente para documentos.")

    # Initialize FAISS index if it doesn't exist
    initialize_faiss_index()

    # Load QA system
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