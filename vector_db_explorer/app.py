from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import sys
import os

# Add vector_db to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db.src import VectorDBClient
from vector_db_explorer.utils.visualizer import ExplorerVisualizer

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Client
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_db', 'chroma_db')
client = VectorDBClient(persist_directory=DB_PATH)
visualizer = ExplorerVisualizer()

@app.route('/', methods=['GET', 'POST'])
def index():
    # 1. Handle Search Logic (if POST)
    query = None
    results = []
    result_indices = [] # Indices of documents that matched
    
    if request.method == 'POST' and 'query' in request.form:
        query = request.form.get('query')
        if query:
            raw_res = client.similarity_search(query, top_k=5)
            if raw_res['documents']:
                docs = raw_res['documents'][0]
                dists = raw_res['distances'][0]
                metas = raw_res['metadatas'][0]
                ids = raw_res['ids'][0]
                
                # We need to map these IDs back to indices for the visualizer
                # This is a bit inefficient (fetching full set to find index) but works for Explorer scale
                all_data = client.get_all_vectors()
                all_ids = all_data['ids']
                
                for i in range(len(docs)):
                    results.append({
                        "id": ids[i],
                        "document": docs[i],
                        "metadata": metas[i],
                        "distance": dists[i]
                    })
                    # Find index in full list
                    if ids[i] in all_ids:
                        result_indices.append(all_ids.index(ids[i]))

    # 2. Fetch Data for Visualizer
    data = client.get_all_vectors()
    plot_html = ""
    doc_count = 0
    
    if data and data['embeddings'] is not None and len(data['embeddings']) > 0:
        doc_count = len(data['embeddings'])
        
        # Calculate Query Vector if needed
        q_vec = None
        if query:
             q_vec = client.encoder.encode([query])[0]
        
        # Generate Plot with Highlights
        plot_html = visualizer.create_plot(
            data['embeddings'], 
            data['documents'], 
            data['metadatas'], 
            query_vector=q_vec,
            search_result_indices=result_indices
        )
    else:
        plot_html = "<div style='padding:20px'>Database is empty.</div>"

    return render_template(
        'index.html', 
        doc_count=doc_count, 
        plot_div=plot_html, 
        results=results, 
        query=query
    )

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        flash("No file part", "error")
        return redirect(url_for('index'))
    
    files = request.files.getlist('files')
    file_paths = []
    
    for file in files:
        if file.filename == '':
            continue
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        file_paths.append(filepath)
    
    if file_paths:
        try:
            client.add_from_files(file_paths)
            flash(f"Successfully processed {len(file_paths)} files.", "success")
        except Exception as e:
            flash(f"Error processing files: {e}", "error")
        
        # Cleanup
        for p in file_paths:
            try:
                os.remove(p)
            except:
                pass
                
    return redirect(url_for('index'))

@app.route('/add_text', methods=['POST'])
def add_text():
    text = request.form.get('text')
    if text:
        client.add_documents([text], metadatas=[{"source": "manual_input"}])
        flash("Text added to vector space.", "success")
    return redirect(url_for('index'))

@app.route('/api/chunks')
def get_chunks():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    query = request.args.get('query', default=None)
    
    offset = (page - 1) * per_page
    
    data, total = client.get_documents(limit=per_page, offset=offset, filter_text=query)
    
    return jsonify({
        "data": data,
        "total": total,
        "page": page,
        "per_page": per_page
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
