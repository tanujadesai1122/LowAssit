import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


# Load the legal cases dataset
data = pd.read_csv("dataset/legal_cases.csv")

# Get case text
case_text = data["case_text"].fillna("")

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

# Convert case text into vectors
case_vectors = vectorizer.fit_transform(case_text)

# Save the vectorizer
joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)

# Save the case vectors
joblib.dump(
    case_vectors,
    "model/case_vectors.pkl"
)

print("===================================")
print("   LAWASSIST MODEL TRAINING")
print("===================================")

print("Total cases:", len(data))
print("Number of features:", len(vectorizer.get_feature_names_out()))

print()
print("TF-IDF model created successfully!")
print("Model files saved inside the model folder.")