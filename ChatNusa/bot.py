import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi stemmer dan stopword
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

# Custom stemmer untuk menangani kata-kata khusus
def custom_stemmer(text):
    kata_khusus = {
        "pengelasan": "ngelas",
        "pembelajaran": "ajar",
        "berbasis": "basis"
    }
    tokens = text.split()
    hasil = [kata_khusus[token] if token in kata_khusus else stemmer.stem(token) for token in tokens]
    return ' '.join(hasil)

# Fungsi preprocessing teks
def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [token for token in tokens if token.isalpha()]
    cleaned = ' '.join(tokens)
    no_stop = stopword.remove(cleaned)
    stemmed = custom_stemmer(no_stop)
    return stemmed

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data Info UNP
with open(os.path.join(BASE_DIR, 'static', 'data', 'info_unp.json'), 'r', encoding='utf-8') as f:
    info_data = json.load(f)

info_df = pd.DataFrame(info_data)
info_df['preprocessed'] = info_df['pertanyaan'].apply(preprocess)

# Load data Repository
with open(os.path.join(BASE_DIR, 'static', 'data', 'repository_data.json'), 'r', encoding='utf-8') as f:
    repo_json = json.load(f)
    repo_data = repo_json[0]['data']

repo_df = pd.DataFrame(repo_data)
repo_df['preprocessed'] = repo_df['title'].apply(preprocess)

# TF-IDF Vectorizer untuk masing-masing dataset
info_vectorizer = TfidfVectorizer()
info_tfidf = info_vectorizer.fit_transform(info_df['preprocessed'])

repo_vectorizer = TfidfVectorizer()
repo_tfidf = repo_vectorizer.fit_transform(repo_df['preprocessed'])

# Fungsi untuk menghitung cosine similarity
def calculate_cosine_scores(query_vector, tfidf_matrix):
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    scores = list(enumerate(similarities))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores

# Fungsi utama sistem pencarian
def search_system(query):
    query_pre = preprocess(query)

    # Vektorisasi query
    info_query_vec = info_vectorizer.transform([query_pre])
    info_scores = calculate_cosine_scores(info_query_vec, info_tfidf)
    best_info_idx, best_info_score = info_scores[0]

    repo_query_vec = repo_vectorizer.transform([query_pre])
    repo_scores = calculate_cosine_scores(repo_query_vec, repo_tfidf)
    top_repo_scores = [score for score in repo_scores if score[1] > 0][:5]

    # Logika pemilihan hasil terbaik
    if best_info_score >= (top_repo_scores[0][1] if top_repo_scores else 0) and best_info_score > 0:
        result = info_df.iloc[best_info_idx]
        return "info", {
            "pertanyaan": result["pertanyaan"],
            "jawaban": result["jawaban"]
        }
    elif top_repo_scores:
        results = []
        for idx, _ in top_repo_scores:
            row = repo_df.iloc[idx]
            results.append({
                "title": row["title"],
                "link": row["link"],
                "year": row.get("year", "Unknown")  # default jika tidak ada
            })
        return "repository", results
    else:
        return "none", None

# Fungsi untuk dipanggil dari Django
def get_response(query):
    return search_system(query)
