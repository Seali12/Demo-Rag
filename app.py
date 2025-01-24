from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from PyPDF2.errors import EmptyFileError, PdfReadError
import os

load_dotenv()

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
        print(f"Error procesando {txt_path}: {str(e)}")
        return ""

def process_folder(folder_path):
    all_text = ""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f"Procesando: {file_path}")
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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    document_search = FAISS.from_texts(texts, embeddings)
    
    os.makedirs("faiss_indexes", exist_ok=True)
    
    document_search.save_local(f"faiss_indexes/{index_name}")
    print(f"Índice guardado como {index_name}")

def load_index(index_name):                                                                      
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    
    return FAISS.load_local(f"faiss_indexes/{index_name}", embeddings, allow_dangerous_deserialization=True)

def mejorar_pregunta(pregunta: str, contexto: str) -> list:
    """Genera alternativas de la misma pregunta de manera más clara o correcta."""
    llm = OllamaLLM(model="llama3", temperature=0.2)

    prompt = f"""
    Eres un asistente especializado en mejorar la redacción de preguntas para hacerlas más claras y correctas.
    Debes considerar que las preguntas deben ser corregidas para que se adapten al español de la Real Academia Española.
    Mantén el significado original de la pregunta y mejora su redacción.
    Evita preguntas ambiguas o confusas.
    Evita proporcionar explicaciones o razones en tu respuesta, y devuelve solo las preguntas reformuladas.
    Tienes acceso al siguiente contexto relacionado con el tema:
    
    {contexto}
    
    Se te proporciona una pregunta escrita por un usuario. 
    Tu tarea es generar al menos 4 versiones de la misma pregunta, mejor redactadas, sin cambiar el significado original 
    y utilizando el contexto cuando sea relevante.
    
    Pregunta original: {pregunta}
    
    Opciones mejoradas:
    """

    result = llm(prompt)
    opciones_mejoradas = result.split("\n")
    return [opcion.strip() for opcion in opciones_mejoradas if opcion.strip()]

def obtener_contexto(index, pregunta, k=5):
    retriever = index.as_retriever(search_kwargs={"k": k})
    resultados = retriever.get_relevant_documents(pregunta)
    contexto = "\n".join([doc.page_content for doc in resultados])
    return contexto

def setup_qa_system(index):
    llm = OllamaLLM(model="llama3", temperature=0)
    prompt_template = """Eres un asistente virtual especializado en los documentos DACA relacionados con el Ejército Argentino. 
    Tu conocimiento abarca reglamentos, procedimientos, jerarquías y normativas específicas de la institución.

    Instrucciones:
    1. Responde siempre en español
    2. No inventes ni asumas información adicional.
    3. Si la información en el contexto es insuficiente para responder completamente, indica claramente qué información adicional sería necesaria.
    4. Organiza tu respuesta de manera clara y estructurada, utilizando viñetas o numeración si es necesario para mejorar la legibilidad.

    Contexto proporcionado:
    {context}

    Pregunta del usuario: {question}

    Respuesta detallada y concisa:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever(search_kwargs={"k": 15}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def ask_question(qa_system, question, index):
    """Realiza una pregunta al sistema de QA y obtiene una respuesta."""
    contexto = obtener_contexto(index, question)
    result = qa_system({"query": question})
    return result["result"], result["source_documents"], contexto

def main():
    folder_path = 'archivos_sanitizados'
    index_name = 'archivos_procesados'

    # Verificar si el índice ya existe
    if not os.path.exists(f"faiss_indexes/{index_name}"):
        print("Procesando archivos...")
        texts = process_folder(folder_path)
        print("Creando índice...")
        create_and_save_index(texts, index_name)
    else:
        print(f"El índice '{index_name}' ya existe. Cargando...")

    # Cargar el índice
    document_search = load_index(index_name)

    # Configurar el sistema de preguntas y respuestas
    qa_system = setup_qa_system(document_search)

    # Interfaz de usuario para preguntas
    print("Sistema de preguntas y respuestas listo. Escribe 'salir' para terminar.")
    while True:
        user_question = input("\nHaz una pregunta sobre los documentos: ")
        if user_question.lower() == 'salir':
            break
        
        # Obtener el contexto y mejorar preguntas
        answer, sources, contexto = ask_question(qa_system, user_question, document_search)
        
        # Generar opciones mejoradas de la pregunta
        opciones_mejoradas = mejorar_pregunta(user_question, contexto)
        
        print("\nRespuesta:", answer)
        
        print("\nOpciones mejoradas de tu pregunta:")
        for i, opcion in enumerate(opciones_mejoradas, 1):
            print(f"{i}. {opcion}")
        
        # Opción para seleccionar una pregunta mejorada
        seleccion = input("\n¿Quieres responder a alguna de estas preguntas? (0 para saltar, número de opción): ")
        try:
            seleccion = int(seleccion)
            if 1 <= seleccion <= len(opciones_mejoradas):
                pregunta_seleccionada = opciones_mejoradas[seleccion - 1]
                print(f"\nPregunta seleccionada: {pregunta_seleccionada}")
                answer, sources, _ = ask_question(qa_system, pregunta_seleccionada, document_search)
                print("\nRespuesta:", answer)
        except ValueError:
            pass

    print("¡Gracias por usar el sistema de preguntas y respuestas!")

if __name__ == "__main__":
    main()