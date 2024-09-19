import streamlit as st
import os
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory



from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")



if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable")
    st.stop()

# Initialize OpenAI model
llm = OpenAI(temperature=0.7, openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize OpenAI's embedding model for document embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Sample documents to showcase FAISS retrieval (replace with your own knowledge base)
documents = [
    "Artificial Intelligence is the future of technology.",
    "Machine Learning enables computers to learn from data.",
    "Data Science is an interdisciplinary field that uses scientific methods to extract knowledge from data.",
    "Natural Language Processing is a key aspect of AI that deals with language understanding."
]

# Convert documents to embeddings and create FAISS index
vectorstore = FAISS.from_texts(documents, embedding_model)

# Set up LangChain RetrievalQA chain with FAISS as retriever
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Set up memory for the conversation
memory = ConversationBufferMemory(memory_key="chat_history")

# Define conversation logic
def chat(user_input):
    # Retrieve relevant documents and respond
    response = qa_chain.run(user_input)
    return response

# Streamlit UI Code
st.title("LangChain Chat Assistant with FAISS")
st.write("Ask anything about AI, Machine Learning, Data Science, or NLP!")

if 'history' not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You: ", "")

# Process input and display chat
if user_input:
    response = chat(user_input)
    st.session_state.history.append((user_input, response))

# Display chat history
if st.session_state.history:
    for user, assistant in st.session_state.history:
        st.write(f"**You**: {user}")
        st.write(f"**Assistant**: {assistant}")
