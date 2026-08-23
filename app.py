from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


# =========================================================
# LOAD LEGAL CASE DATASET
# =========================================================

data = pd.read_csv("dataset/legal_cases.csv")


# =========================================================
# LOAD TF-IDF VECTORIZER
# =========================================================

vectorizer = joblib.load(
    "model/tfidf_vectorizer.pkl"
)


# =========================================================
# LOAD CASE VECTORS
# =========================================================

case_vectors = joblib.load(
    "model/case_vectors.pkl"
)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# SEARCH RELATED CASES
# =========================================================

@app.route("/search", methods=["POST"])
def search():

    # Get case description from the webpage
    query = request.form.get(
        "case_description",
        ""
    ).strip()

    # Check empty input
    if not query:
        return render_template(
            "index.html",
            error="Please enter a case description."
        )

    # =====================================================
    # CONVERT USER CASE INTO TF-IDF VECTOR
    # =====================================================

    query_vector = vectorizer.transform(
        [query]
    )

    # =====================================================
    # CALCULATE COSINE SIMILARITY
    # =====================================================

    similarity_scores = cosine_similarity(
        query_vector,
        case_vectors
    ).flatten()

    # =====================================================
    # SORT CASES FROM HIGHEST TO LOWEST SIMILARITY
    # =====================================================

    sorted_indexes = similarity_scores.argsort()[::-1]

    # =====================================================
    # SELECT CASES WITH AT LEAST 10% SIMILARITY
    # =====================================================

    top_indexes = [
        index
        for index in sorted_indexes
        if similarity_scores[index] >= 0.10
    ][:5]

    # =====================================================
    # CREATE RESULT LIST
    # =====================================================

    results = []

    for index in top_indexes:

        similarity = similarity_scores[index] * 100

        results.append({

            # Dataset row number
            "index": int(index),

            # Case information
            "id": data.iloc[index]["case_id"],

            "title": data.iloc[index]["case_title"],

            "category": data.iloc[index]["category"],

            "court": data.iloc[index]["court"],

            "year": data.iloc[index]["year"],

            # Similarity percentage
            "similarity": round(
                similarity,
                2
            ),

            # Short summary
            "summary": data.iloc[index]["summary"],

            # Complete case text
            "case_text": data.iloc[index]["case_text"]
        })

    # =====================================================
    # SHOW SEARCH RESULTS
    # =====================================================

    return render_template(
        "results.html",
        query=query,
        results=results
    )


# =========================================================
# FULL CASE DETAILS
# =========================================================

@app.route("/case/<int:index>")
def case_details(index):

    # Check whether case exists
    if index < 0 or index >= len(data):

        return "Case not found", 404

    # Get selected case
    case = data.iloc[index]

    # Send case information to HTML page
    return render_template(
        "case_details.html",
        case=case
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )