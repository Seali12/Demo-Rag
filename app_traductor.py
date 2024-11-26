from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

# Cargar las variables de entorno
load_dotenv()

# Procesar archivos desde una carpeta y crear un índice
def crear_indice(carpeta, index_name):
    """Crea un índice a partir de los archivos en la carpeta especificada."""
    print(f"Procesando archivos en la carpeta '{carpeta}'...")

    # Cargar los archivos de texto
    loader = DirectoryLoader(carpeta, glob="*.txt", loader_cls=TextLoader)
    documentos = loader.load()
   
    if not documentos:
        print("No se encontraron archivos en la carpeta especificada.")
        return None

    # Dividir los documentos en fragmentos
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documentos_divididos = splitter.split_documents(documentos)

    # Crear embeddings y construir el índice
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    index = FAISS.from_documents(documentos_divididos, embeddings)

    # Guardar el índice
    os.makedirs("faiss_indexes", exist_ok=True)
    index.save_local(f"faiss_indexes/{index_name}")
    print(f"Índice creado y guardado como '{index_name}'.")
    return index

# Cargar el índice desde disco
def cargar_indice(index_name):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    return FAISS.load_local(f"faiss_indexes/{index_name}", embeddings, allow_dangerous_deserialization=True)

# Obtener contexto relevante del índice
def obtener_contexto(index, pregunta, k=5):
    retriever = index.as_retriever(search_kwargs={"k": k})
    resultados = retriever.get_relevant_documents(pregunta)
    contexto = "\n".join([doc.page_content for doc in resultados])
    return contexto

# Función para generar diferentes formas de hacer la misma pregunta
def mejorar_pregunta(pregunta: str, contexto: str) -> list:
    """Genera alternativas de la misma pregunta de manera más clara o correcta, considerando el contexto."""
    # Define el modelo y el prompt para mejorar las preguntas
    llm = OllamaLLM(model="llama3", temperature=0.2)

    # Crear el prompt con el contexto
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

    # Generar las alternativas
    result = llm(prompt)  # Devuelve un string

    # Dividir las alternativas por saltos de línea
    opciones_mejoradas = result.split("\n")

    # Limpiar líneas vacías o espacios innecesarios
    return [opcion.strip() for opcion in opciones_mejoradas if opcion.strip()]

# Ejemplo de uso de la función
def main():
    carpeta = "archivos"  # Carpeta donde están los archivos
    index_name = "archivos_procesados"  # Nombre del índice

    # Crear el índice si no existe
    if not os.path.exists(f"faiss_indexes/{index_name}"):
        print(f"No se encontró el índice '{index_name}'. Creando uno nuevo...")
        index = crear_indice(carpeta, index_name)
        if not index:
            print("No se pudo crear el índice. Asegúrate de que la carpeta contiene archivos.")
            return
    else:
        print(f"Cargando índice existente '{index_name}'...")
        index = cargar_indice(index_name)

    while True:
        pregunta = input("Introduce una pregunta: ")
        if pregunta.lower() == 'salir':
            break

        # Obtener el contexto relacionado con la pregunta
        contexto = obtener_contexto(index, pregunta)

        if not contexto:
            print("\nNo se encontró contexto relevante para esta pregunta. Asegúrate de que el índice contenga información relacionada.")
            continue

        # Mejora la pregunta original
        opciones_mejoradas = mejorar_pregunta(pregunta, contexto)
       
        print("\nOpciones mejoradas:")
        for i, opcion in enumerate(opciones_mejoradas, 1):
            print(f"{i}. {opcion}")

if __name__ == "__main__":
    main()