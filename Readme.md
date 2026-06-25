\## Resume Clustering and Job Description Matching



\## 📌 Project Overview



This project is helps to recruiter for hiring process. This project helps to rank resume based on keywords.

In this project we can collect resume from candidates and stores it using clustering and matches it from the job description based on keywords.

And suggest the best rank resume based on keywords.



An intelligent Resume Processing System built using Flask, Machine Learning, NLP, FAISS, and Gemini AI.

This project allows you to:



* 📂 Upload multiple resumes
* 🧠 Automatically cluster resumes
* 🏷️ Label clusters using AI
* 📊 Rank resumes based on Job Description
* 🤖 Chat with resumes using an AI-powered chatbot
* 🔄 Retrain chatbot with new resumes



\## 📌 Features

1️⃣ Resume Clustering

Extracts text from resumes

Preprocesses text

Generates embeddings

Applies KMeans clustering

Organizes resumes into cluster folders

Automatically labels clusters using Gemini AI



2️⃣ Resume Ranking

Accepts Job Description input

Compares JD with resume embeddings

Generates similarity-based ranking

Returns structured ranking results



3️⃣ AI Resume Chatbot

Uses FAISS vector database

Retrieves relevant resume chunks

Uses Gemini model to generate contextual responses

Allows interactive Q\&A with uploaded resumes



4️⃣ Retraining Module

Re-parses resumes

Regenerates embeddings

Rebuilds FAISS index

Updates chatbot knowledge base



\## 🛠️ Tech Stack

🐍 Python

🌐 Flask

🤖 SentenceTransformers

📊 Scikit-learn (KMeans)

🔍 FAISS

🧠 Gemini API

🗂️ NumPy

📁 Pathlib \& OS



\## 📂 Project Structure

project/

│

├── app.py

├── input/

│   ├── resumes/

│   └── jd.txt

│

├── output/

│   ├── embeddings.pkl

│   ├── preprocessed\_texts.pkl

│   ├── clusters/

│   ├── resume\_index.faiss

│   └── resume\_metadata.json

│

├── scripts/

│   ├── data\_collection.py

│   ├── text\_extraction.py

│   ├── preprocessing.py

│   ├── feature\_extraction.py

│   ├── clustering.py

│   ├── organize\_clusters.py

│   ├── generate\_summary.py

│   ├── label\_clusters.py

│   ├── rank\_resumes.py

│   └── evaluation.py

│

├── chatbot/

│   ├── parse\_chunk\_resumes.py

│   ├── generate\_embeddings.py

│   ├── store\_embeddings.py

│   ├── retrieve\_chunks.py

│   └── interact\_gemini.py

│

└── templates/



\## 🧠 Machine Learning Details



Embedding Model: all-MiniLM-L6-v2

Clustering: KMeans with optimal K selection

Similarity: Cosine Similarity

Vector Store: FAISS (384-dimension vectors)



\## 👨‍💻 Author



Developed by Gayatri Sanap, Rutik Gunjal, Amit Ghumare, Prajwal Supekar.



