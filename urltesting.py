from flask import Flask, request, render_template, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup
import re

from langchain_text_splitters import HTMLSectionSplitter


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()




app = Flask(__name__)


@app.route('/')
def index():
    return render_template('testing.html')


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
        #print(loader.load())
        result = loader.load()
        #pages = loader.load_and_split(text_splitter=text_splitter)
        context = "\n\n".join([doc.page_content for doc in result])
        print(context)

        return jsonify(context), 200

    except Exception as e:
        return jsonify({'error': 'Failed to process URL'}), 500
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
