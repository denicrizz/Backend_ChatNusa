import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- Inisialisasi ---
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# --- Custom Stemmer ---
def custom_stemmer(text):
    kata_khusus = {
        "pengelasan": "ngelas",
        "pembelajaran": "ajaran",
        "berbasis": "basis",
        "carikan":"cari"
        
    }
    tokens = text.split()
    hasil = [kata_khusus[token] if token in kata_khusus else stemmer.stem(token) for token in tokens]
    return ' '.join(hasil)

# --- Preprocessing ---
def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [token for token in tokens if token.isalpha()]
    clean_text = ' '.join(tokens)
    no_stopword = stopword_remover.remove(clean_text)
    stemmed = custom_stemmer(no_stopword)
    return stemmed

# --- Load JSON dari repository dan info_unp ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    all_data = []

    # Load repository_data.json
    with open(os.path.join(BASE_DIR, 'static', 'data', 'repository_data.json'), 'r', encoding='utf-8') as f:
        repo_json = json.load(f)
        for item in repo_json:
            for entry in item["data"]:
                all_data.append({
                    "title": entry["title"],
                    "link": entry["link"],
                    "source": "repository"
                })

    # Load info_unp.json
    with open(os.path.join(BASE_DIR, 'static', 'data', 'info_unp.json'), 'r', encoding='utf-8') as f:
        unp_json = json.load(f)
        for entry in unp_json:
            all_data.append({
                "title": entry["pertanyaan"],  # gunakan 'pertanyaan' sebagai 'title'
                "link": entry["jawaban"],
                "source": "info_unp"
            })

    return pd.DataFrame(all_data)


# --- Persiapkan Data Sekali ---
df = load_data()
df['preprocessed'] = df['title'].apply(preprocess)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['preprocessed'])
feature_names = vectorizer.get_feature_names_out()

# Di bot.py
def search_repository(query, top_repo=5, top_unp=1):
    # Preproses query normal
    query_processed = preprocess(query)
    
    # Ulang kata-kata penting dalam query untuk meningkatkan pengaruhnya
    # Ini masih menggunakan murni TF-IDF, hanya memanipulasi query
    important_terms = [term for term in query_processed.split() if len(term) > 3]
    emphasized_query = query_processed + " " + " ".join(important_terms)
    
    # Transform ke vektor TF-IDF dan hitung cosine similarity
    query_vector = vectorizer.transform([emphasized_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Proses selanjutnya tetap sama
    df_copy = df.copy()
    df_copy['score'] = similarities
    
    sorted_df = df_copy.sort_values(by='score', ascending=False).drop_duplicates(subset='title')
    
    # Lanjutkan seperti sebelumnya...
    
    # Pisahkan hasil berdasarkan source
    unp_results = sorted_df[sorted_df['source'] == 'info_unp']
    repo_results = sorted_df[sorted_df['source'] == 'repository']
    
    # Ambil top results dari masing-masing kategori
    unp_top = unp_results[unp_results['score'] > 0].head(top_unp)
    repo_top = repo_results[repo_results['score'] > 0].head(top_repo)
    
    # Format hasil untuk info_unp
    info_unp_list = [{
        "title": row["title"],
        "link": row["link"],
        "score": float(row["score"])
    } for _, row in unp_top.iterrows()]
    
    # Format hasil untuk repository
    repository_list = [{
        "title": row["title"],
        "link": row["link"],
        "score": float(row["score"])
    } for _, row in repo_top.iterrows()]
    
    return info_unp_list, repository_list

def detect_intent(query):
    repo_keywords = ["carikan","skripsi", "judul", "tugas akhir", "game", "pemrograman", "aplikasi", "android", "sistem", "metode"]
    info_unp_keywords = ["syarat", "aturan", "jadwal", "biaya", "dosen", "akreditasi", "alamat", "kontak"]

    query_lower = query.lower()
    repo_count = sum([kw in query_lower for kw in repo_keywords])
    info_count = sum([kw in query_lower for kw in info_unp_keywords])

    if repo_count > info_count:
        return "repository"
    else:
        return "info_unp"
    
# --- Fungsi yang Dipanggil dari Django View ---
def get_response(query, top_repo=5, top_unp=1, threshold=0.3):
    info_unp_results, repo_results = search_repository(query, top_repo=top_repo, top_unp=top_unp)
    
    # Jika tidak ada hasil sama sekali
    if not info_unp_results and not repo_results:
        return "no_result", None
    
    # Cek apakah ada hasil info_unp yang cukup relevan
    if info_unp_results and info_unp_results[0]['score'] > threshold:
        # Return format untuk info_unp
        return "info", {
            "pertanyaan": info_unp_results[0]['title'],
            "jawaban": info_unp_results[0]['link']
        }
    
    
    # Jika tidak ada hasil info_unp yang relevan, kembalikan hasil repository
    if repo_results:
        # Return format untuk repository
        return "repository", repo_results
    
    # Jika info_unp tidak cukup relevan dan tidak ada repository, kembalikan hasil info_unp
    if info_unp_results:
        return "info", {
            "pertanyaan": info_unp_results[0]['title'],
            "jawaban": info_unp_results[0]['link']
        }
    
    return "no_result", None


