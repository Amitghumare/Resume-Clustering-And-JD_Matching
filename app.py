from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, send_file
import os, shutil
from pathlib import Path
import json, time
import numpy as np
from pathlib import Path
# Import your project pipeline functions
from scripts.clustering import clustering_pipeline
from scripts.evaluation import evaluation_pipeline
from scripts.generate_summary import generate_summary
from scripts.rank_resumes import rank_resumes
from chatbot.interact_gemini import load_environment_variables, initialize_gemini, generate_response


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Define key folder paths
INPUT_RESUMES_FOLDER = os.path.join('input', 'resumes')
OUTPUT_FOLDER = 'output'
JD_FILE = os.path.join('input', 'jd.txt')

# Ensure necessary folders exist
Path(INPUT_RESUMES_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    if request.method == 'POST':
        # Use only the base filename (ignore folder structure)
        uploaded_files = request.files.getlist("folder_files")
        copied_files = []
        for file in uploaded_files:
            filename = os.path.basename(file.filename)
            dest_path = os.path.join(INPUT_RESUMES_FOLDER, filename)
            if not os.path.exists(dest_path):
                file.save(dest_path)
                copied_files.append(filename)
        flash(f"Copied {len(copied_files)} new file(s) to the input folder.")
        return redirect(url_for('clustering'))
    return render_template('clustering.html')

@app.route('/start_clustering', methods=['GET'])
def start_clustering():
    def generate():
        progress = []
        try:
            # --- Step 1: Data Collection ---
            from scripts.data_collection import get_resume_files
            resume_files = get_resume_files(INPUT_RESUMES_FOLDER)
            msg = f"Data Collection: Collected {len(resume_files)} resume file(s) from input/resumes."
            progress.append(msg)
            yield f"data: {json.dumps({'progress': msg})}\n\n"
            time.sleep(1)

            # --- Step 2: Text Extraction ---
            from scripts.text_extraction import extract_texts
            extracted_texts = extract_texts(resume_files)
            msg = f"Text Extraction: Extracted text from {len(extracted_texts)} resume(s)."
            progress.append(msg)
            yield f"data: {json.dumps({'progress': msg})}\n\n"
            time.sleep(1)

            # --- Step 3: Preprocessing ---
            from scripts.preprocessing import preprocess_texts, save_preprocessed_texts
            preprocessed_texts = preprocess_texts(extracted_texts)
            preprocessed_texts_path = os.path.join(OUTPUT_FOLDER, 'preprocessed_texts.pkl')
            save_preprocessed_texts(preprocessed_texts, preprocessed_texts_path)
            msg = f"Preprocessing: Preprocessed {len(preprocessed_texts)} resume(s)."
            progress.append(msg)
            yield f"data: {json.dumps({'progress': msg})}\n\n"
            time.sleep(1)

            # --- Step 4: Feature Extraction ---
            from scripts.feature_extraction import feature_extraction
            embeddings = feature_extraction(preprocessed_texts, save_path=os.path.join(OUTPUT_FOLDER, 'embeddings.pkl'))
            msg = f"Feature Extraction: Generated embeddings for {len(embeddings)} resume(s)."
            progress.append(msg)
            yield f"data: {json.dumps({'progress': msg})}\n\n"
            time.sleep(1)

            # --- Step 5: Clustering ---
            from scripts.clustering import clustering_pipeline
            embeddings_path = os.path.join(OUTPUT_FOLDER, 'embeddings.pkl')
            assignments_path = os.path.join(OUTPUT_FOLDER, 'cluster_assignments.pkl')
            cluster_assignments, kmeans_model = clustering_pipeline(embeddings_path, save_path=assignments_path, max_k=10)
            msg = "Clustering: Completed clustering and determined optimal number of clusters."
            progress.append(msg)
            yield f"data: {json.dumps({'progress': msg})}\n\n"
            time.sleep(1)

            # --- Step 6: Organize Clusters ---
            from scripts.organize_clusters import organize_clusters
            clusters_output_folder = os.path.join(OUTPUT_FOLDER, 'clusters')
            organize_clusters(assignments_path, embeddings_path, output_folder=clusters_output_folder)
            msg = "Organizing Clusters: Resumes organized into cluster folders."
            progress.append(msg)
            yield f"data: {json.dumps({'progress': msg})}\n\n"
            time.sleep(1)

            # --- Step 7: Generate Summary ---
            md_summary, html_summary = generate_summary(assignments_path, preprocessed_texts_path, output_folder=OUTPUT_FOLDER, summary_filename='cluster_summary.md')
            msg = "Generating Summary: Cluster summary generated."
            progress.append(msg)
            yield f"data: {json.dumps({'progress': msg})}\n\n"
            time.sleep(1)

            # --- Step 8: Label Clusters ---
            from scripts.label_clusters import label_clusters
            clusters_info_path = os.path.join(OUTPUT_FOLDER, 'clusters_info.json')
            label_clusters(clusters_info_path, clusters_output_folder, output_folder=OUTPUT_FOLDER, model_name="gemini-1.5-flash")
            msg = "Labeling Clusters: Clusters labeled successfully."
            progress.append(msg)
            yield f"data: {json.dumps({'progress': msg})}\n\n"
            time.sleep(1)

            # Read clusters from output/clusters folder
            clusters_folder = os.path.join(OUTPUT_FOLDER, 'clusters')
            clusters_dict = {}
            if os.path.exists(clusters_folder):
                for folder in os.listdir(clusters_folder):
                    folder_path = os.path.join(clusters_folder, folder)
                    if os.path.isdir(folder_path):
                        clusters_dict[folder] = os.listdir(folder_path)
            else:
                clusters_dict = {}

            final_data = {
                'progress': progress,
                'summary': html_summary,  # Use attractive HTML summary
                'clusters': clusters_dict
            }
            yield f"data: {json.dumps({'final': final_data})}\n\n"
        except Exception as e:
            err_msg = "Error: " + str(e)
            yield f"data: {json.dumps({'progress': err_msg, 'error': True})}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<cluster>/<filename>')
def download(cluster, filename):
    clusters_folder = os.path.join(OUTPUT_FOLDER, 'clusters')
    file_path = os.path.join(clusters_folder, cluster, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404
    
@app.route('/download_resume/<filename>')
def download_resume(filename):
    file_path = os.path.join(INPUT_RESUMES_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404





# @app.route('/ranking', methods=['GET', 'POST'])
# def ranking():
#     if request.method == 'POST':
#         jd_text = request.form.get('jd_text')
#         if not jd_text:
#             flash("Please enter a Job Description.")
#             return redirect(url_for('ranking'))
#         with open(JD_FILE, 'w', encoding='utf-8') as f:
#             f.write(jd_text)
#         rank_resumes(
#             jd_path=JD_FILE,
#             embeddings_path=os.path.join(OUTPUT_FOLDER, 'embeddings.pkl'),
#             preprocessed_texts_path=os.path.join(OUTPUT_FOLDER, 'preprocessed_texts.pkl'),
#             output_folder=OUTPUT_FOLDER,
#             summary_filename='jd_rankings.md'
#         )
#         with open(os.path.join(OUTPUT_FOLDER, 'jd_rankings.md'), 'r', encoding='utf-8') as f:
#             rankings_content = f.read()
#         return render_template('ranking.html', rankings=rankings_content)
#     return render_template('ranking.html', rankings=None)



@app.route('/ranking', methods=['GET', 'POST'])
def ranking():
    # We will store the final ranking results in 'rankings'
    rankings = None

    if request.method == 'POST':
        jd_text = request.form.get('jd_text', '').strip()

        # If JD text is provided, you can handle it by writing to a temporary file,
        # or you can modify your rank_resumes function to accept raw text directly.
        # For simplicity, let's assume we write to a temporary .txt file:
        jd_path = 'temp_jd.txt'
        with open(jd_path, 'w', encoding='utf-8') as f:
            f.write(jd_text)

        # Import your rank_resumes function
        from scripts.rank_resumes import rank_resumes

        # Call rank_resumes with return_data=True to get structured results
        results = rank_resumes(
            jd_path=jd_path,
            embeddings_path='output/embeddings.pkl',
            preprocessed_texts_path='output/preprocessed_texts.pkl',
            output_folder='output',
            summary_filename='jd_rankings.md',
            return_data=True
        )
        rankings = results

        # (Optional) remove or keep the temp file if needed
        # os.remove(jd_path)

    return render_template('ranking.html', rankings=rankings)

@app.route('/upload_files', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist("files")
    copied_files = []
    for file in uploaded_files:
        filename = os.path.basename(file.filename)
        dest_path = os.path.join(INPUT_RESUMES_FOLDER, filename)
        if not os.path.exists(dest_path):
            file.save(dest_path)
            copied_files.append(filename)
    return jsonify({'status': 'success', 'copied_files': copied_files})

@app.route('/list_files', methods=['GET'])
def list_files():
    file_list = os.listdir(INPUT_RESUMES_FOLDER)
    return jsonify({'files': file_list})

@app.route('/remove_file', methods=['POST'])
def remove_file():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'status': 'error', 'message': 'Filename not provided.'})
    file_path = os.path.join(INPUT_RESUMES_FOLDER, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({'status': 'success', 'message': f"'{filename}' removed."})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    else:
        return jsonify({'status': 'error', 'message': 'File not found.'})
    


# Updated Chatbot Route using custom logic (from main_chatbot.py)
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        data = request.get_json()
        user_query = data.get('message', '')
        if not user_query:
            return jsonify({'error': 'No message provided.'})
        try:
            from chatbot.interact_gemini import load_environment_variables, initialize_gemini, generate_response_with_chunks
            from chatbot.retrieve_chunks import retrieve_chunks
        except Exception as e:
            return jsonify({'error': str(e)})
        
        # Set paths to FAISS index and metadata (adjust as needed)
        project_root = Path(__file__).parent.resolve()
        index_path = project_root / "output" / "resume_index.faiss"
        metadata_path = project_root / "output" / "resume_metadata.json"
        if not index_path.exists() or not metadata_path.exists():
            return jsonify({'error': 'Index or metadata file not found. Please run store_embeddings.py first.'})
        
        try:
            load_environment_variables()
            gemini_model = initialize_gemini()
        except Exception as e:
            return jsonify({'error': str(e)})
        
        retrieved_chunks = retrieve_chunks(
            query=user_query,
            index_path=index_path,
            metadata_path=metadata_path,
            model_name='all-MiniLM-L6-v2',
            top_k=5
        )
        if not retrieved_chunks:
            return jsonify({'error': 'No relevant information found in resumes.'})
        response_text = generate_response_with_chunks(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            model=gemini_model
        )
        return jsonify({'response': response_text})
    else:
        return render_template('chatbot.html')


# # New route to retrain the chatbot model with updated data
# @app.route('/retrain', methods=['POST'])
# def retrain():
#     try:
#         # Step 1: Parse and chunk resumes
#         from chatbot.parse_chunk_resumes import parse_and_chunk_resumes
#         input_folder = Path("input/resumes")
#         output_chunks = Path("output/resume_chunks.json")
#         # You can adjust max_length as needed
#         parse_and_chunk_resumes(input_folder, output_chunks, max_length=500)
        
#         # Step 2: Generate embeddings for the resume chunks
#         from chatbot.generate_embeddings import generate_embeddings, load_model
#         model = load_model()  # Load the SentenceTransformer model
#         input_file = output_chunks  # This is the JSON with resume chunks
#         output_embeddings = Path("output/resume_embeddings.json")
#         generate_embeddings(input_file, output_embeddings, model)
        
#         # Step 3: Build (or update) the FAISS index and metadata
#         from chatbot.store_embeddings import store_embeddings
#         index_output = Path("output/resume_index.faiss")
#         metadata_output = Path("output/resume_metadata.json")
#         store_embeddings(input_file, index_output, metadata_output, dimension=384)
        
#         return jsonify({"status": "success", "message": "Retraining complete. The chatbot model has been updated with the new data."})
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)})
@app.route('/retrain', methods=['GET'])
def retrain():
    def generate():
        try:
            from chatbot.parse_chunk_resumes import parse_and_chunk_resumes
            from chatbot.generate_embeddings import load_model, generate_embeddings
            from chatbot.store_embeddings import store_embeddings
            from time import sleep

            # Step 1: Parse and chunk resumes
            input_folder = Path("input/resumes")
            output_chunks = Path("output/resume_chunks.json")
            parse_and_chunk_resumes(input_folder, output_chunks, max_length=500)
            yield f"data: {json.dumps({'progress': 'Step 1 completed: Parsed and chunked resumes'})}\n\n"
            sleep(1)

            # Step 2: Load the model
            model = load_model()
            yield f"data: {json.dumps({'progress': 'Step 2 completed: Model loaded'})}\n\n"
            sleep(1)

            # Step 3: Generate embeddings
            input_file = output_chunks
            output_embeddings = Path("output/resume_embeddings.json")
            generate_embeddings(input_file, output_embeddings, model)
            yield f"data: {json.dumps({'progress': 'Step 3 completed: Embeddings generated'})}\n\n"
            sleep(1)

            # Step 4: Build FAISS index
            index_output = Path("output/resume_index.faiss")
            metadata_output = Path("output/resume_metadata.json")
            store_embeddings(input_file, index_output, metadata_output, dimension=384)
            yield f"data: {json.dumps({'progress': 'Step 4 completed: FAISS index updated'})}\n\n"
            sleep(1)

            # Finish
            final_message = "Retraining complete. The chatbot model has been updated with the new data."
            yield f"data: {json.dumps({'final': final_message})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
