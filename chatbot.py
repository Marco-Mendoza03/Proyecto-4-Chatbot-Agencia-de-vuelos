from os import getenv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import gradio as gr

load_dotenv()

# Configurar modelo de lenguaje con OpenRouter y Helicone
llm = ChatOpenAI(
  openai_api_key=getenv("OPENROUTER_API_KEY"),
  openai_api_base=getenv("OPENROUTER_BASE_URL"),
  model_name="openai/gpt-4o",
  model_kwargs={
    "extra_headers":{
        "Helicone-Auth": f"Bearer "+getenv("HELICONE_API_KEY")
      }
  },
)

# Mensaje del sistema: contextualizar el chatbot para una agencia de vuelos
system_message = SystemMessage(
    content="Eres un asistente virtual de una agencia de vuelos. Ayudas a los usuarios respondiendo típicas cuestiones sobre reservas de vuelos, políticas de equipaje, reservas, cancelaciones y otros servicios relacionados con vuelos."
)

# Función del chatbot
def chatbot(message, history):
    # Preparar el historial con el mensaje del sistema
    messages = [system_message]
    
    # Agregar el historial previo al contexto
    for user_message, bot_message in history:
        messages.append(HumanMessage(content=user_message)) #https://python.langchain.com/docs/concepts/messages/#humanmessage
        messages.append(SystemMessage(content=bot_message)) #https://python.langchain.com/docs/concepts/messages/#systemmessage
    
    # Agregar el mensaje actual del usuario
    messages.append(HumanMessage(content=message)) # se va juntando todo en la misma lista
    
    # Procesar la respuesta en streaming
    response = llm.stream(messages)
    partial_response = ""
    for chunk in response:
        if chunk and hasattr(chunk, 'content'):
            content = chunk.content
            if content is not None:
                partial_response += content
                yield partial_response

    
# Crear la interfaz del chatbot
demo = gr.ChatInterface(
    chatbot,
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Escribe tu mensaje aquí...", container=False, scale=7),
    title="Asistente para Agencia de Vuelos",
    description="Haz tus consultas sobre vuelos: equipaje, cambios, cancelaciones, y más.",
    theme="soft",
    examples=[
        "¿Cuánto cuesta cambiar un vuelo?",
        "¿Qué equipaje puedo llevar en un vuelo nacional?",
        "¿Cuál es la política de cancelación?",
        "¿Puedo elegir asiento después de reservar?",
        "¿Cómo puedo añadir equipaje extra?",
    ],
    type="messages",
    editable=True,
    save_history=True,
)

# Lanzar la aplicación
if __name__ == "__main__":
    demo.queue().launch()
