from flask import Flask, request, render_template, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup
import re
from langchain_text_splitters import HTMLHeaderTextSplitter


from langchain_text_splitters import HTMLSectionSplitter

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
    return render_template('testing.html')


@app.route('/geturl', methods=['POST'])
def geturl():
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        print(url)
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
        ]
        
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        html_header_splits = html_splitter.split_text_from_url(url)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
        )
        result =text_splitter.split_documents(html_header_splits)
        
        print(result)
        
        
        #context = "\n\n".join([doc.page_content for doc in pages])
        #html_header_splits = html_splitter.split_text(context)
        #print(html_header_splits)
        #resurlt = "\n\n".join([doc.page_content for doc in html_header_splits])


        return jsonify({'message': f'URL uploaded successfully'}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to process URL'}), 500
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
