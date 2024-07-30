from langchain_community.chat_message_histories import SQLChatMessageHistory
from flask import Flask, request, render_template, jsonify
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import HTMLHeaderTextSplitter
import re

# Code for debuging
chat_history = SQLChatMessageHistory(
    session_id="1", connection_string="sqlite:///sqlite.db"
)
#    messages = chain_with_history.messages
#    print(messages)

#Define Model
llm = ChatOllama(model="qwen2:0.5b")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a bot your name is Alice you should reply in 100 words or less"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{context}\n\nQ: {question}\nA:"),
    ]
)

chain = prompt | llm


chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sqlite.db"
    ),
    input_messages_key="question",
    history_messages_key="history",
)


chroma_db = 'chromadb'

embedding_function = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

db = Chroma(persist_directory=chroma_db, embedding_function=embedding_function)

# Enhanced extractor function
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for script in soup(["script", "style"]):
        script.decompose()  # Remove all script and style elements
    text = soup.get_text()  # Extract the text content
    text = re.sub(r"\n\s*\n", "\n\n", text)  # Remove multiple newlines
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('chat3.html')



@app.route('/query', methods=['POST'])
def query():
    try: 
        data = request.get_json()
        query_text = data.get("query_text", "")
        print("question: " ,query_text, "\n\n")

        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.1,
                },
        )
        documents = retriever.invoke(query_text)
        print(documents)
        context = "\n\n".join([doc.page_content for doc in documents])
        #print("context: ", context, "\n\n")
        config = {"configurable": {"session_id": "1"}}
        response = chain_with_history.invoke({"question": query_text, "context": context }, config=config)
        #print("response: ", response.content, "\n\n")
        
        #messages = chat_history.messages
        #print("messages: ", messages)
        return jsonify({'message': response.content}), 200

    except Exception as e:
        print(e)
        return jsonify({'message': 'Failed to process'}), 500



@app.route('/geturl', methods=['POST'])
def geturl():
    try:
        data = request.get_json()
        url = data.get('url')
        print(url)
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        loader = RecursiveUrlLoader(url, extractor=bs4_extractor)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
        )

        pages = loader.load_and_split(text_splitter=text_splitter)

        Chroma.from_documents(pages, embedding_function, persist_directory=chroma_db)

        return jsonify({'message': f'URL uploaded successfully'}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to process URL'}), 500
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
