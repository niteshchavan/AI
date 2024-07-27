from langchain_community.chat_message_histories import SQLChatMessageHistory
from flask import Flask, request, render_template, jsonify
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
import re
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize chat message history
session_id = "1"
connection_string = "sqlite:///sqlite.db"

chat_message_history = SQLChatMessageHistory(
    session_id=session_id, connection_string=connection_string
)



# Define Model
llm = ChatOllama(model="qwen2:0.5b")

# Create Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a bot your name is Alice you should reply in 100 words or less"), 
    MessagesPlaceholder(variable_name="history"),
    ("human", "{context}\n\nQ: {question}\nA:")
    ]
)

#Memory Management
def get_session_history(session_id):
    print(session_id)
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")
    
runnable_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history,
)

# Create Chain with History
chain = prompt_template | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string=connection_string
    ),
    input_messages_key="question",
    history_messages_key="history",
)

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

chroma_db = 'chromadb'

embedding_function = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

db = Chroma(persist_directory=chroma_db, embedding_function=embedding_function)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat3.html')




@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get("query_text", "")
    print(query_text)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
            search_kwargs={
                "k": 2,
                "score_threshold": 0.1,
            },
    )
    relevant_documents = retriever.invoke(query_text)
    
    results = "\n\n".join([doc.page_content for doc in relevant_documents])

    formatted_prompt = prompt_template.format(context=results, question=query_text)
    response = runnable_with_history.invoke([HumanMessage(content=formatted_prompt)],config={"configurable": {"session_id": "1"}},)
    messages = chat_message_history.messages
    print(messages)

    return jsonify({'message': response.content}), 200
    


@app.route('/geturl', methods=['POST'])
def geturl():
    try:
        data = request.get_json()
        url = data.get('url')
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
        logger.error(f"Error processing URL: {e}")
        return jsonify({'error': 'Failed to process URL'}), 500



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
