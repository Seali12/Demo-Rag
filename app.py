from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from PyPDF2.errors import EmptyFileError, PdfReadError
import os
from multiprocessing import Pool, cpu_count
load_dotenv() #ejecuta lo q esta en env
def process_pdf(pdf_path):
    try:
        pdfReader = PdfReader(pdf_path)
        texto_crudo = ''
        for page in pdfReader.pages:
            contenido = page.extract_text()
            if contenido:
                #  diferentes codificaciones
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        texto_crudo += contenido.encode(encoding).decode('utf-8')
                        break
                    except UnicodeDecodeError:
                        continue
        return texto_crudo
    except EmptyFileError:
        print(f"Warning: {pdf_path} is empty. Skipping.")
        return ""
    except PdfReadError:
        print(f"Warning: Could not read {pdf_path}. Skipping.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred with {pdf_path}: {str(e)}. Skipping.")
        return ""

def process_folder(folder_path):
    all_text = ""
    for root,dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                print(f"Procesando: {pdf_path}")
                file_content = process_pdf(pdf_path)
                if file_content:
                    all_text += file_content + "\n\n"
    
    if not all_text:
        raise ValueError("No valid PDF content found in the specified folder.")
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000, #pruebo un valor mas alto
        chunk_overlap=800,
        length_function=len,
    )
    return text_splitter.split_text(all_text)



def create_and_save_index(texts, index_name):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2" ) #max_seq_length=512
    document_search = FAISS.from_texts(texts, embeddings)
    
    os.makedirs("faiss_indexes", exist_ok=True)
    
    document_search.save_local(f"faiss_indexes/{index_name}")
    print(f"Índice guardado como {index_name}")

def load_index(index_name):                                                                         # aca decia cpu
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"} )#max_seq_length=512
    return FAISS.load_local(f"faiss_indexes/{index_name}", embeddings,allow_dangerous_deserialization=True)


def setup_qa_system(index):

    llm = Ollama(model="llama2")
    #probar ponerle que es de recursos humanos
    prompt_template = """Eres un asistente virtual especializado en los documentos REDOAPE relacionados con el Ejército Argentino. Tu conocimiento abarca reglamentos, procedimientos, jerarquías y normativas específicas de la institución.

        Instrucciones:
        1. Responde siempre en español, utilizando terminología oficial del Ejército Argentino cuando sea apropiado.
        2. Basa tus respuestas únicamente en la información proporcionada en el contexto. No inventes ni asumas información adicional.
        3. Si la información en el contexto es insuficiente para responder completamente, indica claramente qué partes de la pregunta puedes responder y qué información adicional sería necesaria.
        4. Si encuentras información contradictoria en el contexto, señálalo en tu respuesta y explica las discrepancias.
        5. Si la pregunta involucra aspectos legales o administrativos, cita la normativa o reglamento relevante si está disponible en el contexto.
        6. Organiza tu respuesta de manera clara y estructurada, utilizando viñetas o numeración si es necesario para mejorar la legibilidad.

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
        retriever=index.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


def ask_question(qa_system, question):
    """Realiza una pregunta al sistema de QA y obtiene una respuesta."""
    result = qa_system({"query": question})
    return result["result"], result["source_documents"]

### Paso 5: Configuración principal ###
def main():
    folder_path = 'Anexo3'  # Reemplaza esto con la ruta a tu carpeta de PDFs
    index_name = 'archivos_procesados'

    # Verificar si el índice ya existe
    if not os.path.exists(f"faiss_indexes/{index_name}"):
        print("Procesando archivos PDF...")
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
        
        answer, sources = ask_question(qa_system, user_question)
        print("\nRespuesta:", answer)

    print("¡Gracias por usar el sistema de preguntas y respuestas!")

# Ejecutar el programa principal
if __name__ == "__main__":
    main()