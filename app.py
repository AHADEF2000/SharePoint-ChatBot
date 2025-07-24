import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDFs
from docx import Document as DocxDocument
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Step 1: Extract text from PDF and Word documents
def extract_text_from_files(folder_path="docs"):
    texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            try:
                doc = fitz.open(file_path)
                text = "\n".join([page.get_text() for page in doc])
                texts.append(Document(page_content=text, metadata={"source": filename}))
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
        elif filename.endswith(".docx"):
            try:
                doc = DocxDocument(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
                texts.append(Document(page_content=text, metadata={"source": filename}))
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
    return texts

# Step 2: Chunk the documents
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Step 3: Create a vector database
def create_vector_db(chunks):
    return Chroma.from_documents(chunks, OpenAIEmbeddings())

# Step 4: Build LangChain QA pipeline with memory
def build_conversational_chain(vector_db):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=vector_db.as_retriever(),
        memory=memory
    )

# Step 5: Prepare the chatbot
print("🔄 Loading and indexing documents...")
docs = extract_text_from_files("docs")
chunks = chunk_documents(docs)
vector_db = create_vector_db(chunks)
qa_chain = build_conversational_chain(vector_db)
chat_history = []

print("✅ Chatbot is ready.")

# Step 6: Set up Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    user_input = request.json.get("question", "")
    print(f"🧑 User asked: {user_input}")

    if not user_input.strip():
        return jsonify({"answer": "❗يرجى كتابة سؤال."})

    try:
        result = qa_chain.invoke({
            "question": user_input,
            "chat_history": chat_history
        })
        answer = result["answer"]
        chat_history.append((user_input, answer))
        print(f"🤖 Bot replied: {answer}")
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"❌ Error during QA chain: {e}")
        return jsonify({"answer": "حدث خطأ أثناء الإجابة. حاول مرة أخرى."})

if __name__ == "__main__":
    app.run(debug=True)
