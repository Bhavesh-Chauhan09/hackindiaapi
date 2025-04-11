from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

def extract_keywords_tfidf(query, top_n=5):
    # Add a dummy document for TF-IDF vectorizer to work
    documents = [query, ""]

    # TF-IDF Vectorizer with n-grams (1 to 3 words)
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(documents)

    feature_array = np.array(tfidf.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray()[0]
    tfidf_sorting = np.argsort(tfidf_scores)[::-1]

    top_keywords = feature_array[tfidf_sorting][:top_n]
    return top_keywords.tolist()

@app.route('/extract_keywords', methods=['POST'])
def extract():
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing "query" parameter in JSON body'}), 400

    query = data['query']
    top_n = data.get('top_n', 5)  # Optional: number of keywords

    keywords = extract_keywords_tfidf(query, top_n)

    return jsonify({'keywords': keywords})

if __name__ == '__main__':
    app.run(debug=True)
