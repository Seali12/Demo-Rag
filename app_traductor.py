from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# ---- Funciones comunes para ambos sistemas ----

# Crear un índice a partir de documentos
def crear_indice(carpeta, index_name):
    print(f"Procesando archivos en la carpeta '{carpeta}'...")
    loader = DirectoryLoader(
        carpeta,
        glob="*.txt",
        loader_cls=lambda file_path: TextLoader(file_path, encoding="utf-8")
    )
    documentos = loader.load()
    if not documentos:
        print("No se encontraron archivos en la carpeta especificada.")
        return None

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documentos_divididos = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    index = FAISS.from_documents(documentos_divididos, embeddings)

    os.makedirs("faiss_indexes", exist_ok=True)
    index.save_local(f"faiss_indexes/{index_name}")
    print(f"Índice creado y guardado como '{index_name}'.")
    return index

# Cargar índice desde disco
def cargar_indice(index_name):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    return FAISS.load_local(f"faiss_indexes/{index_name}", embeddings, allow_dangerous_deserialization=True)

# Obtener contexto relevante del índice
def obtener_contexto(index, pregunta, k=5):
    retriever = index.as_retriever(search_kwargs={"k": k})
    resultados = retriever.get_relevant_documents(pregunta)
    contexto = "\n".join([doc.page_content for doc in resultados])
    return contexto

# ---- Funciones del sistema de mejora de preguntas ----

def mejorar_pregunta(pregunta: str, contexto: str) -> list:
    llm = OllamaLLM(model="llama3", temperature=0.2)
    prompt = f"""
    Eres un asistente especializado en mejorar la redacción de preguntas para hacerlas más claras y correctas.
    Debes considerar que las preguntas deben ser corregidas para que se adapten al español de la Real Academia española.
    Debes mantener el significado original de la pregunta y mejorar su redacción.
    Debes evitar preguntas ambiguas o confusas.
    Debes reemplazar apodos, por sinonimos o términos más formales.
    Debes evitar preguntas que puedan tener respuestas subjetivas.
    Tienes acceso al siguiente contexto relacionado con el tema:
    
    {contexto}
    
    A continuación, se te proporciona una pregunta escrita por un usuario. Tu tarea es generar al menos 4 versiones de la misma pregunta, mejor redactadas, sin cambiar el significado original y utilizando el contexto cuando sea relevante.
    
    Pregunta original: {pregunta}
    
    Opciones mejoradas:
    """
    result = llm(prompt)
    opciones = []
    for linea in result.split("\n"):
        linea = linea.strip()
        if linea.startswith("¿") and linea.endswith("?"):
            opciones.append(linea)
    opciones.append(pregunta)  # Agregar la pregunta original como opción
    return opciones

# ---- Configuración del sistema de preguntas y respuestas ----

def setup_qa_system(index):
    llm = OllamaLLM(model="llama3", temperature=0.1)
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
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever(search_kwargs={"k": 10}),
        chain_type_kwargs={"prompt": PROMPT}
    )

# ---- Flujo principal ----

def main():
    carpeta = "archivos_sanitizados"  # Carpeta donde están los archivos
    index_name = "archivos_procesados"

    # Crear o cargar el índice
    if not os.path.exists(f"faiss_indexes/{index_name}"):
        print(f"No se encontró el índice '{index_name}'. Creando uno nuevo...")
        index = crear_indice(carpeta, index_name)
        if not index:
            print("No se pudo crear el índice. Asegúrate de que la carpeta contiene archivos.")
            return
    else:
        print(f"Cargando índice existente '{index_name}'...")
        index = cargar_indice(index_name)

    qa_system = setup_qa_system(index)

    while True:
        pregunta = input("Introduce una pregunta: ")
        if pregunta.lower() == 'salir':
            break

        # Obtener el contexto relacionado con la pregunta
        contexto = obtener_contexto(index, pregunta)

        if not contexto:
            print("\nNo se encontró contexto relevante para esta pregunta.")
            continue

        # Mejora la pregunta original
        opciones_mejoradas = mejorar_pregunta(pregunta, contexto)

        print("\nOpciones mejoradas:")
        for i, opcion in enumerate(opciones_mejoradas, 1):
            print(f"{i}. {opcion}")

        seleccion = int(input("\nElige el número de la pregunta que deseas (o escribe 0 para cancelar): "))
        if seleccion == 0:
            continue

        # Obtener la pregunta elegida
        try:
            pregunta_elegida = opciones_mejoradas[seleccion - 1]
            print(f"\nProcesando la pregunta elegida: {pregunta_elegida}")
        except IndexError:
            print("\nSelección no válida. Intenta nuevamente.")
            continue

        # Pasar la pregunta elegida al sistema de QA
        respuesta, fuentes = qa_system.invoke({"query": pregunta_elegida})
        print("\nRespuesta:", respuesta)

if __name__ == "__main__":
    main()
