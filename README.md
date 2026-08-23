# ⚖ LawAssist - AI-Based Legal Case Research Assistant

LawAssist is an AI/ML-based legal research application that helps users find
legal cases related to a given case description.

The user enters a short description of a legal case, and the system uses
Natural Language Processing (NLP) and TF-IDF with Cosine Similarity to find
the most relevant cases from the legal case dataset.

---

## 🚀 Features

- 🔍 Search for related legal cases
- 🤖 AI/ML-based case similarity
- 📚 Find multiple related cases
- 📊 Similarity percentage for each case
- 🏛 Display court and case category
- 📅 Display case year
- 📖 View complete case details
- 📝 View case summary
- 💻 Simple web interface
- ⚡ Lightweight and suitable for low-system computers

---

## 🧠 How It Works

The system follows these steps:

1. User enters a legal case description.
2. The description is converted into a TF-IDF vector.
3. The system compares the user description with the legal cases
   in the dataset.
4. Cosine Similarity is calculated.
5. Cases are ranked according to similarity.
6. The most relevant cases are displayed.
7. User can open a case to study its complete details.

---

## 🏗 System Architecture

```text
User
  │
  ▼
Case Description
  │
  ▼
Flask Web Application
  │
  ▼
Text Preprocessing
  │
  ▼
TF-IDF Vectorization
  │
  ▼
Cosine Similarity
  │
  ▼
Legal Case Dataset
  │
  ▼
Related Legal Cases
  │
  ▼
Full Case Details