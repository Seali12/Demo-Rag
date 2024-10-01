import os
from concurrent.futures import ProcessPoolExecutor
from PyPDF2 import PdfReader
from PyPDF2.errors import EmptyFileError, PdfReadError
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from llama_index.core import SimpleDirectoryReader, ServiceContext
from llama_index.core import  GPTVectorStoreIndex
from llama_index.legacy import  LLMPredictor
from langchain.chains.retrieval_qa.base import RetrievalQA
from llama_index.legacy.embeddings import LangchainEmbedding
from llama_index.legacy.storage import StorageContext
from llama_index.legacy import  load_index_from_storage

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

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
    """Processes all PDFs in the folder and returns them as a list of text chunks."""
    all_text = []
    with ProcessPoolExecutor() as executor:
        for root, dirs, files in os.walk(folder_path):
            pdf_paths = [os.path.join(root, file) for file in files if file.endswith('.pdf')]
            for file_content in executor.map(process_pdf, pdf_paths):
                if file_content:
                    all_text.append(file_content)
    
    if not all_text:
        raise ValueError("No valid PDF content found in the specified folder.")
    
    return all_text

def create_and_save_index(texts, index_name):
    """Creates and saves a LlamaIndex from the provided texts."""
    # Convert texts into documents
    documents = [Document(text) for text in texts]

    # Set up the LLM predictor using Ollama
    llm_predictor = LLMPredictor(llm=Ollama(model="llama3"))

    # Set up embeddings using HuggingFace
    embeddings = LangchainEmbedding(HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    # Create a service context
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embeddings)

    # Create the index
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    # Save the index
    index.storage_context.persist(f"./indexes/{index_name}")
    print(f"Index saved as {index_name}")

def load_index(index_name):
    """Loads a saved LlamaIndex index."""
    storage_context = StorageContext.from_defaults(persist_dir=f"./indexes/{index_name}")
    return load_index_from_storage(storage_context)

def setup_qa_system(index):
    """Sets up the QA system using the specified LlamaIndex."""
    llm = Ollama(model="llama3")
    
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
        retriever=index.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

if __name__ == '__main__':
    folder_path = 'AlgunosAnexos'
    index_name = 'IndiceMultiplesPDFs'

    # Check if the index exists, otherwise create it
    if not os.path.exists(f"./indexes/{index_name}"):
        print("Processing PDF files...")
        texts = process_folder(folder_path)
        print("Creating index...")
        create_and_save_index(texts, index_name)
    else:
        print("Index already exists. Loading...")

    # Load the index and set up the QA system
    index = load_index(index_name)
    qa_system = setup_qa_system(index)

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
