#pip install langchain
#pip install openai
#pip install pypdf2
#pip install faiss-cpu
#pip install tiktoken
#pip install InstructorEmbedding
#pip install sentence_transformers
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
 #probar embeddins de hugging face
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.ollama import Ollama

#pip install langchain_community
import os
load_dotenv() #ejecuta lo q esta en env
#Si tira error del token utilizar sentence-transformers==2.2.2
def process_pdf(pdf_path):
    pdfReader = PdfReader(pdf_path)
    texto_crudo = ''
    for page in pdfReader.pages:
        contenido = page.extract_text()
        if contenido:
            texto_crudo += contenido
    return texto_crudo

def process_folder(folder_path):
    all_text = ""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                print(f"Procesando: {pdf_path}")
                all_text += process_pdf(pdf_path) + "\n\n"
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(all_text)

def create_and_save_index(texts, index_name):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",encode_kwargs={"clean_up_tokenization_spaces": True}
                                               ) #max_seq_length=512
    document_search = FAISS.from_texts(texts, embeddings)
    
    os.makedirs("faiss_indexes", exist_ok=True)
    
    document_search.save_local(f"faiss_indexes/{index_name}")
    print(f"Índice guardado como {index_name}")

def load_index(index_name):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                                model_kwargs={"device": "cpu"}
                                                )#max_seq_length=512
    return FAISS.load_local(f"faiss_indexes/{index_name}", embeddings, allow_dangerous_deserialization=True)

def setup_qa_system(index):
    llm = Ollama(model="llama3")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain

def ask_question(qa_system, question):
    result = qa_system({"query": question})
    return result["result"], result["source_documents"]

# Configuración principal
folder_path = 'Anexo1'  # Reemplaza esto con la ruta a tu carpeta de PDFs
index_name = 'IndiceMultiplesPDFs'

# Verificar si el índice ya existe
if not os.path.exists(f"faiss_indexes/{index_name}"):
    print("Procesando archivos PDF...")
    texts = process_folder(folder_path)
    print("Creando índice...")
    create_and_save_index(texts, index_name)
else:
    print("El índice ya existe. Cargando...")

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
    print("\nFuentes:")
    for i, doc in enumerate(sources, 1):
        print(f"Fuente {i}:", doc.page_content[:100], "...")
    print("\n" + "-"*50)

print("¡Gracias por usar el sistema de preguntas y respuestas!")