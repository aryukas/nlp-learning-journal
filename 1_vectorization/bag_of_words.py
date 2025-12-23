from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# -------------------------------
# Sample corpus
# -------------------------------
corpus = [
    "NLP logic learning",
    "NLP is amazing and powerful",
    "Learning new NLP concepts in detail"
    "I am Aryan Kashyap"
]

# -------------------------------
# Basic Bag of Words
# -------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Convert to DataFrame
bow_df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("Bag of Words Matrix:\n")
print(bow_df)

# -------------------------------
# Vocabulary
# -------------------------------
print("\nVocabulary:\n")
print(vectorizer.vocabulary_)

# -------------------------------
# Word Frequency
# -------------------------------
word_counts = bow_df.sum(axis=0)
print("\nWord Frequencies:\n")
print(word_counts)

# -------------------------------
# BoW with Stopwords Removed
# -------------------------------
vectorizer_sw = CountVectorizer(stop_words='english')
X_sw = vectorizer_sw.fit_transform(corpus)

bow_df_sw = pd.DataFrame(
    X_sw.toarray(),
    columns=vectorizer_sw.get_feature_names_out()
)

print("\nBoW with Stopwords Removed:\n")
print(bow_df_sw)

# -------------------------------
# BoW with N-grams
# -------------------------------
vectorizer_ngram = CountVectorizer(ngram_range=(1, 2))
X_ngram = vectorizer_ngram.fit_transform(corpus)

bow_df_ngram = pd.DataFrame(
    X_ngram.toarray(),
    columns=vectorizer_ngram.get_feature_names_out()
)

print("\nBoW with Unigrams + Bigrams:\n")
print(bow_df_ngram)
