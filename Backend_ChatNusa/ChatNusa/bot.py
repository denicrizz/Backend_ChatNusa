import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi stemmer dan stopword
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [token for token in tokens if token.isalpha()]
    cleaned = ' '.join(tokens)
    no_stop = stopword.remove(cleaned)
    stemmed = stemmer.stem(no_stop)
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
info_features = info_vectorizer.get_feature_names_out()

repo_vectorizer = TfidfVectorizer()
repo_tfidf = repo_vectorizer.fit_transform(repo_df['preprocessed'])
repo_features = repo_vectorizer.get_feature_names_out()

def calculate_score(query_tokens, tfidf_matrix, features, vectorizer):
    scores = []
    for i in range(tfidf_matrix.shape[0]):
        doc_vec = tfidf_matrix[i].toarray().flatten()
        doc_score = 0.0
        for token in query_tokens:
            if token in features:
                idx = vectorizer.vocabulary_.get(token)
                if idx is not None:
                    doc_score += doc_vec[idx]
        scores.append((i, doc_score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores

def search_system(query):
    query_pre = preprocess(query)
    query_tokens = query_pre.split()

    info_scores = calculate_score(query_tokens, info_tfidf, info_features, info_vectorizer)
    best_info_idx, best_info_score = info_scores[0]

    repo_scores = calculate_score(query_tokens, repo_tfidf, repo_features, repo_vectorizer)
    top_repo_scores = [score for score in repo_scores if score[1] > 0][:5]  # ambil 5 teratas

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
