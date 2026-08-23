import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# Load the dataset
data = pd.read_csv("dataset/legal_cases.csv")

# Load the trained TF-IDF model
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Load the case vectors
case_vectors = joblib.load("model/case_vectors.pkl")


print("===================================")
print("       LAWASSIST CASE SEARCH")
print("===================================")

# Get case description from lawyer
query = input("\nEnter your case description:\n")

# Convert lawyer's description into TF-IDF vector
query_vector = vectorizer.transform([query])

# Calculate similarity with all cases
similarity_scores = cosine_similarity(
    query_vector,
    case_vectors
).flatten()

# Get indexes of top 5 similar cases
top_indexes = similarity_scores.argsort()[-5:][::-1]


print("\n===================================")
print("       RELATED CASES")
print("===================================")

for rank, index in enumerate(top_indexes, start=1):

    similarity = similarity_scores[index] * 100

    print(f"\n{rank}. {data.iloc[index]['case_title']}")
    print(f"   Category: {data.iloc[index]['category']}")
    print(f"   Court: {data.iloc[index]['court']}")
    print(f"   Year: {data.iloc[index]['year']}")
    print(f"   Similarity: {similarity:.2f}%")
    print(f"   Summary: {data.iloc[index]['summary']}")

print("\n===================================")