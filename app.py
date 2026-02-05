from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt   # FIXED import

# ------------------ Flask setup ------------------
app = Flask(__name__)
load_dotenv()

# ------------------ Environment variables ------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    raise EnvironmentError("Missing API keys in .env file")

# ------------------ Embeddings ------------------
embeddings = download_hugging_face_embeddings()

# ------------------ Pinecone Vector Store ------------------
index_name = "medical-chatbot"

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ------------------ LLM ------------------
llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
)

llm = ChatHuggingFace(llm=llm_endpoint)

# ------------------ Prompt ------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# ------------------ LCEL RAG CHAIN ------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------ Routes ------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    print("User:", user_input)

    response = rag_chain.invoke(user_input)

    print("Bot:", response)
    return response

# ------------------ Run app ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
