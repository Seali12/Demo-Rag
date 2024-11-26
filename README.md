# Sistema de Preguntas y Respuestas basado en PDFs

Este proyecto implementa un sistema de preguntas y respuestas que utiliza documentos PDF como fuente de conocimiento. El sistema procesa múltiples archivos PDF, crea un índice searchable, y permite a los usuarios hacer preguntas sobre el contenido de estos documentos.

## Características

- Procesa múltiples archivos PDF en una carpeta especificada
- Crea un índice FAISS para búsqueda eficiente
- Utiliza embeddings de Hugging Face para la representación del texto
- Implementa un sistema de preguntas y respuestas utilizando el modelo Llama 3 a través de Ollama
- Interfaz de línea de comandos para interactuar con el sistema

## Requisitos

- Python 3.7+
- Bibliotecas requeridas (ver `requirements.txt`)
- crear un archivo .env en el cual contiene HUGGINGFACEHUB_API_TOKEN

## Instalación

1. Clona este repositorio:
2. Instala las dependencias
3. Asegúrate de tener Ollama instalado y el modelo Llama 3 descargado.

## Uso

1. Coloca tus archivos PDF en la carpeta `Anexo1` (o modifica la variable `folder_path` en el script).

2. Ejecuta el script
3. El sistema procesará los PDFs y creará un índice la primera vez que se ejecute.

4. Una vez cargado, puedes hacer preguntas sobre el contenido de los documentos.

5. Escribe 'salir' para terminar la sesión.

## Configuración
- Instalar Ollama y elegir el modelo llama3, ejecutando en la consola de Ollama el comando ollama pull llama3.
- Descargar Python.
- Descargar las librerias del archivo requirements.txt con el comando pip install -r requirements.txt.
- Si el dispositivo cuenta cuenta nucleos cuda cambiar en la funcionj load_index en la variable embeddings cpu por cuda, descargar de la pagina pytorch -> get started -> elegir el pip que da la pagina.
  
## Notas

- Este sistema utiliza FAISS para el almacenamiento y búsqueda de vectores, lo que permite una recuperación rápida de información relevante.
- El sistema no actualiza el índice automáticamente si se añaden nuevos PDFs. Deberás eliminar manualmente el índice existente para recrearlo con nuevos documentos.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de enviar un pull request.

## Licencia

[Incluye aquí la información de la licencia de tu proyecto]
