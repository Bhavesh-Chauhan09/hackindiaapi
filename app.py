from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask TF-IDF Keyword Extractor is running!"

def extract_keywords_tfidf(query):
    # Add a dummy document to enable TF-IDF computation
    documents = [query, ""]

    # Create TF-IDF vectorizer with 1-2 word n-grams
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(documents)

    # Get feature names and scores for the actual query
    feature_array = np.array(tfidf.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Filter out zero-score keywords
    nonzero_indices = tfidf_scores.nonzero()[0]
    filtered_features = feature_array[nonzero_indices]
    filtered_scores = tfidf_scores[nonzero_indices]

    # Sort by score (descending)
    sorted_indices = np.argsort(filtered_scores)[::-1]
    sorted_keywords = filtered_features[sorted_indices]

    return sorted_keywords.tolist()

@app.route('/extract_keywords', methods=['POST'])
def extract():
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({'error': 'Missing "query" parameter in JSON body'}), 400

    query = data['query']
    keywords = extract_keywords_tfidf(query)

    return jsonify({'keywords': keywords})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
