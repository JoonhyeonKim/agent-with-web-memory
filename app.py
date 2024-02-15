import os
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import initialize_agent
import gradio as gr
from langchain.globals import set_debug
set_debug(True)

# load the env variables
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY =os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID =os.getenv("GOOGLE_CSE_ID")

# then set the tools 
search = DuckDuckGoSearchResults()
tools = [
    Tool(
    name ="Search" ,
    func=search.run,
    description="useful when you need to answer questions about current events"
    ),
]
memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True) # one time memory for the convs

llm=ChatOpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description",verbose=True, memory=memory)

# or like this: prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Answer all questions to the best of your ability.",
#         ),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#     ]
# )
def chat_response(input_text):
    response=agent_chain.run(input=input_text)
    return response
interface = gr.Interface(fn=chat_response, inputs="text", outputs="text", description="Chat with a conversational agent")
interface.launch(share=True)