import os
from concurrent.futures import ProcessPoolExecutor
from PyPDF2 import PdfReader
from PyPDF2.errors import EmptyFileError, PdfReadError
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def process_pdf(pdf_path):
    """Extracts text from a PDF, handling errors gracefully."""
    try:
        pdfReader = PdfReader(pdf_path)
        return "\n".join([page.extract_text() for page in pdfReader.pages if page.extract_text()])
    except (EmptyFileError, PdfReadError) as e:
        print(f"Warning: Could not process {pdf_path}: {str(e)}. Skipping.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred with {pdf_path}: {str(e)}. Skipping.")
        return ""

def process_folder(folder_path):
    """Processes all PDFs in the folder and splits text into chunks."""
    all_text = []
    with ProcessPoolExecutor() as executor:
        for root, dirs, files in os.walk(folder_path):
            pdf_paths = [os.path.join(root, file) for file in files if file.endswith('.pdf')]
            for file_content in executor.map(process_pdf, pdf_paths):
                if file_content:
                    all_text.append(file_content)
    
    if not all_text:
        raise ValueError("No valid PDF content found in the specified folder.")
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1600, #subir a 2k
        chunk_overlap=400,
        length_function=len,
    )
    return text_splitter.split_text("\n\n".join(all_text))

def create_and_save_index(texts, index_name):
    """Creates and saves a FAISS index for the provided texts."""
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #probar otro emebdding

    document_search = FAISS.from_texts(texts, embeddings)
    
    os.makedirs("faiss_indexes", exist_ok=True)
    document_search.save_local(f"faiss_indexes/{index_name}")
    print(f"Index saved as {index_name}")

def load_index(index_name):
    """Loads a saved FAISS index."""
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(f"faiss_indexes/{index_name}", embeddings, allow_dangerous_deserialization=True)

def setup_qa_system(index):
    """Sets up the QA system using the specified FAISS index."""
    llm = Ollama(model="llama3") #modelo a evaluar
    
    prompt_template = """Eres un asistente virtual especializado en temas legales, civiles y administrativos relacionados con el Ejército Argentino. Tu conocimiento abarca reglamentos, procedimientos, jerarquías y normativas específicas de la institución.

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
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever(search_kwargs={"k": 8}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def ask_question(qa_system, question, conversation_history):
    """Asks a question to the QA system, including conversation history."""
    
    # Format the conversation history into the context
    history_context = "\n".join([f"Pregunta: {q}\nRespuesta: {a}" for q, a in conversation_history])

    # Ask the new question, passing the history as context
    result = qa_system({"query": question, "context": history_context})
    
    # Add the new question and answer to the history
    conversation_history.append((question, result["result"]))
    
    return result["result"], result["source_documents"]

if __name__ == '__main__':
    folder_path = 'AlgunosAnexos'
    index_name = 'IndiceMultiplesPDFs'

    # Check if the index exists, otherwise create it
    if not os.path.exists(f"faiss_indexes/{index_name}"):
        print("Processing PDF files...")
        texts = process_folder(folder_path)
        print("Creating index...")
        create_and_save_index(texts, index_name)
    else:
        print("Index already exists. Loading...")

    # Load the index and set up the QA system
    document_search = load_index(index_name)
    qa_system = setup_qa_system(document_search)

    # Initialize the conversation history
    conversation_history = []

    # User Interface for asking questions
    print("QA system ready. Type 'salir' to quit.")
    while True:
        user_question = input("\nAsk a question about the documents: ")
        if user_question.lower() == 'salir':
            break

        # Ask question and include conversation history
        answer, sources = ask_question(qa_system, user_question, conversation_history)
        
        # Display the answer and sources
        print("\nAnswer:", answer)
        print("\nSources:")
        for i, doc in enumerate(sources, 1):
            print(f"Source {i}:", doc.page_content[:100], "...")
        print("\n" + "-"*50)

    print("Thank you for using the QA system!")