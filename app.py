from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
#from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM
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

def process_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            texto_crudo = ''
            for line in file:
                # Try different encodings if needed
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        texto_crudo += line.encode(encoding).decode('utf-8')
                        break
                    except UnicodeDecodeError:
                        continue
        return texto_crudo
    except FileNotFoundError:
        print(f"Warning: {txt_path} does not exist. Skipping.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred with {txt_path}: {str(e)}. Skipping.")
        return ""



# def process_folder(folder_path):
#     all_text = ""
#     for root,dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith('.pdf'):
#                 pdf_path = os.path.join(root, file)
#                 print(f"Procesando: {pdf_path}")
#                 file_content = process_pdf(pdf_path)
#                 if file_content:
#                     all_text += file_content + "\n\n"
            
def process_folder(folder_path):
    all_text = ""
    for root,dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                pdf_path = os.path.join(root, file)
                print(f"Procesando: {pdf_path}")
                file_content = process_txt(pdf_path)
                if file_content:
                    all_text += file_content + "\n\n"
                


    if not all_text:
        raise ValueError("No valid PDF content found in the specified folder.")
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400, # 400
        chunk_overlap=100,# 100
        length_function=len,
    )
    print(text_splitter)
    return text_splitter.split_text(all_text)



def create_and_save_index(texts, index_name):#probar otro embedding on mas tokens
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2" ) #max_seq_length=512
    document_search = FAISS.from_texts(texts, embeddings)
    
    os.makedirs("faiss_indexes", exist_ok=True)
    
    document_search.save_local(f"faiss_indexes/{index_name}")
    print(f"Índice guardado como {index_name}")

def load_index(index_name):                                                                         # aca decia cpu
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"} )#max_seq_length=512
    return FAISS.load_local(f"faiss_indexes/{index_name}", embeddings,allow_dangerous_deserialization=True)


def setup_qa_system(index):

    llm = OllamaLLM(model="llama3")
    #probar ponerle que es de recursos humanos
    prompt_template = """Eres un asistente virtual especializado en los documentos REDOAPE relacionados con el Ejército Argentino. 
    Tu conocimiento abarca reglamentos, procedimientos, jerarquías y normativas específicas de la institución.

        Instrucciones:
        1. Responde siempre en español
        2. Basa tus respuestas únicamente en la información proporcionada en el contexto. No inventes ni asumas información adicional.
        3. Si la información en el contexto es insuficiente para responder completamente, indica claramente qué información adicional sería necesaria.
        4. Si encuentras información contradictoria en el contexto, señálalo en tu respuesta y explica las discrepancias.
        5. Organiza tu respuesta de manera clara y estructurada, utilizando viñetas o numeración si es necesario para mejorar la legibilidad.

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
        retriever=index.as_retriever(search_kwargs={"k": 3}),
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
    folder_path = 'ArchivosDemotxt'  # Reemplaza esto con la ruta a tu carpeta de PDFs
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






#     ¿Qué requisitos debe cumplir un postulante para ser incorporado como Oficial del Escalafón de Sistema de Computación de Datos?
# Respuesta: Ser argentino/a, de estado civil indistinto, tener hasta 29 años de edad, 
# poseer título de analista de sistemas de nivel terciario o superior (mínimo 3 años de estudios), 
# no poseer antecedentes penales, aprobar el examen psicofísico, presentar currículum vitae, y tener 
# domicilio legal en la localidad de la Guarnición donde se incorporará.






# Cuales son los requisitos de edad para la incorporacion de suboficiales al escalafon de Baqueanos?


# Respuesta: **Análisis y respuesta**

# La pregunta se refiere a los requisitos de edad para la incorporación de suboficiales al escalafón de Baqueanos. 
# Para responder, debemos analizar ambos documentos proporcionados.       

# En el documento LAFÓN DE BAQUEANOS, se establece que para ser Soldado Voluntario en una Unidad o Subunidad Independiente de Montaña o Monte,
# los requisitos de incorporación son: ser argentino nativo o por opción, de sexo masculino, estado civil indistinto y haber aprobado las exigencias 
# correspondientes al IIdo Ciclo del Subplan de capacitación del SMV. No se menciona edad específica en este documento.

# Por otro lado, el APENDICE 3a AL ANEXO 3, RECLUTAMIENTO LOCAL DE OFICIALES DEL ESCALAFÓN DE SISTEMA DE COM-PUTACIÓN DE DATOS 
# (OEJEMGE Nro 1010/01 – SISTEMA DE INCORPORACIÓN SIGLO XXI) establece los requisitos para la incorporación de oficiales, 
# incluyendo suboficiales. En este documento, se especifica que el requisito de edad es tener hasta VEINTINUEVE (29) años de edad a la 
# fecha de presentación de la solicitud.

# **Conclusión**

# En cuanto a los requisitos de edad para la incorporación de suboficiales al escalafón de Baqueanos, podemos concluir que:

# * No se menciona edad específica en el documento LAFÓN DE BAQUEANOS.
# * En el APENDICE 3a AL ANEXO 3, RECLUTAMIENTO LOCAL DE OFICIALES DEL ESCALAFÓN DE SISTEMA DE COM-PUTACIÓN DE DATOS, 
# se establece que los suboficiales deben tener hasta VEINTINUEVE (29) 
# años de edad a la fecha de presentación de la solicitud.

# **Nota**

# Es importante mencionar que si existiera algún requisito adicional o modificación en otros documentos o normativas no 
# proporcionados, no podemos hacer referencia a ellos en nuestra respuesta.









#  departamento es responsable de preparar y corregir los exámenes para el cambio de escalafón?

# Respuesta: La respuesta a esta pregunta se puede encontrar en la normativa específica del Ejército Argentino relacionada con el personal "En Comisión".

# Según lo establecido en el Reglamento de Escalafones del Ejército Argentino, artículo 12.3, es responsabilidad del Departamento de Personal (DP) 
# la preparación y corrección de los exámenes para el cambio de escalafón.

# Además, en el Procedimiento Operativo Estándar (POE) N° 001/17, se indica que el DP es el encargado de elaborar y corregir los exámenes para 
# la evaluación del personal "En Comisión" que solicita cambiar de escalafón.

# Es importante destacar que en este contexto, no hay información adicional necesaria para responder completamente a esta pregunta.


#se confuncde baja con reincorporacion



#  me quiero dar de baja voluntaria, que tengo que hacer?

# Entiendo que deseas darte de baja voluntariamente. Para proceder con la solicitud, te diré lo siguiente:

# De acuerdo al contexto proporcionado, no hay un procedimiento específico mencionado para dar de baja voluntariamente.
# Sin embargo, como se refiere a "Solicitud de Permuta" en el párrafo segundo, asumo que la solicitud de baja voluntaria 
# se realizará a través del sistema de Declaración Jurada (DUFI) y electrónico (EE), similar al procedimiento para revalidar un expediente.

# **Responsabilidades**

# 1. Como causante, debes:
#         * Realizar una Declaración Jurada en el DUFI y EE, indicando tus intenciones de darse de baja voluntariamente.
#         * Responsabilidad del Jefe del Elemento: verificar la situación y aprobar o desaprobar la solicitud.

# **Recomendación**

# Antes de realizar la solicitud, es importante que revisen las normas y procedimientos aplicables a su caso específico.
# Como asistente virtual especializado en documentos REDOAPE relacionados con el Ejército Argentino, puedo recomendar que
# consulten directamente con un oficial o funcionario competente para obtener orientación y guía sobre el proceso de solicitud de baja voluntaria.

# **Nota**

# Es importante mencionar que, según el contexto proporcionado, no hay un plazo específico mencionado para darse de baja voluntariamente.
# Sin embargo, es recomendable realizar la solicitud lo antes posible para evitar posibles problemas administrativos o legales.

# Haz una pregunta sobre los documentos: salir
# ¡Gracias por usar el sistema de preguntas y respuestas!




#con txt
# Haz una pregunta sobre los documentos: cual es el procedimiento para otorgar autorizacion para rei


# Respuesta: La respuesta al procedimiento para otorgar autorización para reincorporar a un oficial
# TORGAR AUTORIZACIÓN PARA REINCORPORACIÓN DE PERSONAL DE OFICIALES Y SUBOFICIALES.

# Según este procedimiento, el personal de Oficiales/Suboficiales que haya sido dado de baja a su sompre y cuando
# lo haga antes de transcurridos dos años desde la fecha de su baja (punto 2.a).

# Además, el causante deberá someterse a un reconocimiento médico (punto 2.a). El procedimiento no ente su solicitud
# y se somete al reconocimiento médico.

# Sin embargo, en el punto 3), se menciona que con la resolución adoptada por el Director de Persona
# Es posible que este sea el siguiente paso después de la solicitud y el reconocimiento médico.

# En resumen, el procedimiento para otorgar autorización para reincorporar a un oficial implica presentar la solicitud antes 
# de transcurridos dos años desde la fecha de su baja y someterse a un reconocimiento médico. Luego, se remite el expediente 
# al Dpto Reg e Inf para posteriores trámites con la resolución adoptada por el Director de Personal Militar.