# Inicios
¿Cómo empezar?
## Instalar las librerías necesarias.


## Definir el caso.
Crear un chatbot de apoyo ante el cliente o los trabajadores de una agencia de vuelos.

## Definir funciones básicas.
- RAG: Consultas sobre políticas y preguntas frecuentes desde una base de datos
- Agente: Realizar una tarea como calcular un precio con fechas concretas y clase de vuelo

## Organizar la estructura del proyecto
- Empezar con la construcción del chatbot
- Mejorar el chatbot (extenderlo)


## ¿Qué es el procesamiento del lenguaje natural?
Es el campo de conocimiento de la Inteligencia Artificial que se ocupa de la investigar la manera de comunicar las máquinas con las personas mediante el uso de lenguas naturales.
En el caso de este chatbot, se usa para la comprensión de los prompt que escribe el usuario, usando información relevante de los documentos aportados y generando respuestas con sentido.

## Utilidad del PLN en el chatbot
Permite interpretar preguntas formuladas por el usuario en un lenguaje común.
Usa embeddings y vectores para usarlo a la hora de buscar información en el documento
Utiliza la API Tavily Search, un motor de búsqueda específico para agentes de IA, buscando resultados para responder a las preguntas eficientemente.
Y finalmente, intenta responder en streaming, ya que lo hace solo al usar el RAG, pero cuando usa el agente no lo hace.

## Modelo de lenguaje utilizado
Se usa GPT-4o de OpenAI mediante Open Router. Además se usa:

- Hugging Face embeddings: con el modelo sentence-transformers/all-mpnet-base-v2. Sirve para generar representaciones vectoriales con los textos. En pocas palabras, busca fragmentos en el documento que sean relevantes conforme a las preguntas del usuario.

- FAISS: Se complemente con el anterior. El primero divide en fragmentos y convierte estos en vectores numéricos, y estos se guardan en FAISS, una base de datos. Cuando se hace una pregunta, FAISS busca los fragmentos más similares al vector de la pregunta y los recupera.

- LangChain: Para la creación del agente con herramientas como Tavily Search

- LangGraph: Implementación de flujos conversacionales estructurados (Diseño de bot para que la conversación siga una estructura predefinida con pasos bien definidos)

### Diagrama en mermaid:

    graph TD;
    
    A["Usuario hace una pregunta"] -->|¿Contiene 'ley' o 'artículo'?| B["Usa RAG para buscar fragmentos relevantes"]
    
    A -->|Pregunta general| C["Usa agente con herramientas de navegación"]
    
    B --> D["Devuelve respuesta basada en documentos"]
    
    C --> E["Interacciona con herramientas externas"]
    
    E --> F["Devuelve respuesta basada en análisis externo"]
    
    D --> G["Respuesta mostrada al usuario"]
    
    F --> G
