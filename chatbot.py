from os import getenv
from dotenv import load_dotenv
import gradio as gr
from gradio import ChatMessage
from langchain_openai import ChatOpenAI

# Para el RAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Para el agente
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Inicializamos el modelo de ChatOpenAI
llm = ChatOpenAI(
    openai_api_key=getenv("OPENROUTER_API_KEY"),
    openai_api_base=getenv("OPENROUTER_BASE_URL"),
    model_name="openai/gpt-4o",
    model_kwargs={
        "extra_headers": {
            "Helicone-Auth": f"Bearer " + getenv("HELICONE_API_KEY")
        }
    },
)

# Cargamos el documento de la ley 21/2003, de 7 de julio, de Seguridad Aérea
url = "https://www.boe.es/buscar/act.php?id=BOE-A-2003-13616"
loader = WebBaseLoader(
    web_paths=(url,),
)
docs = loader.load()

# Dividimos el texto en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Inicializamos Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Creamos un vector store con FAISS
vectorstore = FAISS.from_documents(splits, embeddings)

# Configuración del agente
search_tool = TavilySearchResults(max_results=2)
memory = MemorySaver()
agent_executor = create_react_agent(
    llm,
    tools=[search_tool],
    checkpointer=memory
)


# Función principal del chatbot (responde en streaming y muestra en consola los trozos usados)
def chatbot(message, history):
    
    messages_for_agent = []
    for msg in history:
        if msg["role"] == "user":
            messages_for_agent.append(HumanMessage(content=msg["content"]))
        else:
            messages_for_agent.append(AIMessage(content=msg["content"]))
    
    
    
    # Agregamos el mensaje actual del usuario
    messages_for_agent.append(HumanMessage(content=message))
    
    # Este es el único filtro que he usado para elegir entre usar RAG o el agente
    if "ley" in message.lower() or "articulo" in message.lower():
        print("\n=== Fragmentos de documento utilizados para la respuesta ===")
        relevant_docs = vectorstore.similarity_search(message)
        for i, doc in enumerate(relevant_docs, 1):
            texto = doc.page_content.replace("\n", " ")
            print(f"\nFragmento {i}:\n{texto[:300]}...")
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        final_prompt = (
            "Eres un asistente experto en la ley 21/2003 de Seguridad Aérea. "
            "Utiliza el siguiente contexto para responder de forma breve y concisa.\n\n"
            f"Contexto:\n{context_text}\n\n"
            f"Pregunta: {message}\n"
            "Respuesta:"
        )
        
        messages = []
        for chat_message in history:
            messages.append(chat_message)
        messages.append({"role": "user", "content": final_prompt})

        response = llm.stream(messages)
        partial_response = ""
        for chunk in response:
            if chunk and hasattr(chunk, "content"):
                content = chunk.content
                if content is not None:
                    partial_response += content
                    yield partial_response
    else:
        messages_for_agent = []
        for msg in history:
            if msg["role"] == "user":
                messages_for_agent.append(HumanMessage(content=msg["content"]))
            else:
                messages_for_agent.append(AIMessage(content=msg["content"]))

        # Agregamos el mensaje actual del usuario
        messages_for_agent.append(HumanMessage(content=message))

        # Creamos la lista local para ir añadiendo mensajes formateados (ChatMessage) para Gradio
        streamed_history = []
        # Devolvemos la primera versión para que Gradio inicie el render
        yield streamed_history

        # Invocamos al agente en modo streaming.
        # Importante: pasamos config con un thread_id para la memoria
        config = {"configurable": {"thread_id": "demo-thread"}}

        for chunk in agent_executor.stream({"messages": messages_for_agent}, config=config):
            # Si hay mensajes de tipo herramienta
            if "tools" in chunk:
                for tool_msg in chunk["tools"]["messages"]:
                    # Todo el texto, aviso + resultado, en un único ChatMessage
                    thinking_content = (
                        "Necesito usar la herramienta Tavily...\n\n"
                        f"Resultado de Tavily:\n{tool_msg.content}"
                    )
                    thinking_msg = ChatMessage(
                        role="assistant",
                        content=thinking_content,
                        metadata={"title": "Buscando 🔎"}
                    )
                    streamed_history.append(thinking_msg)
                    yield streamed_history

            # Si hay respuesta del agente (texto final)
            if "agent" in chunk:
                for agent_msg in chunk["agent"]["messages"]:
                    final_msg = ChatMessage(
                        role="assistant",
                        content=agent_msg.content
                    )
                    streamed_history.append(final_msg)
                    yield streamed_history

# Interfaz de Gradio
demo = gr.ChatInterface(
    chatbot,
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Escribe tu mensaje aquí...", container=False, scale=7),
    title="ChatBot RAG + Agente - Ley seguridad Aérea",
    description="Asistente virtual experto en la ley 21/2003 de 7 de julio de Seguridad Aérea .",
    theme="ocean",
    examples=[
        "Articulo 3. Programa Nacional de Seguridad para la Aviación Civil.",
        "Explica la Coordinación entre los Ministerios de Defensa y de Fomento sobre la ley de Seguridad Aérea.",
        "¿Cuál es el articulo 21?"
    ],
    type="messages",
    editable=True,
    save_history=True,
)

if __name__ == "__main__":
    demo.queue().launch()
