from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
corpus = [
    "I love natural language processing",
    "NLP is amazing and powerful",
    "I love learning new NLP concepts"
]
vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(corpus)
tfidf_df = pd.DataFrame(
    X_tfidf.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("TF-IDF Matrix:\n")
tfidf_df
print("Vocabulary:\n")
vectorizer.vocabulary_
#scores
tfidf_scores = tfidf_df.sum(axis=0).sort_values(ascending=False)

print("TF-IDF Scores (Higher = More Important):\n")
tfidf_scores
vectorizer_sw = TfidfVectorizer(stop_words='english')

X_sw = vectorizer_sw.fit_transform(corpus)

tfidf_df_sw = pd.DataFrame(
    X_sw.toarray(),
    columns=vectorizer_sw.get_feature_names_out()
)

print("TF-IDF without Stopwords:\n")
tfidf_df_sw
vectorizer_ngram = TfidfVectorizer(ngram_range=(1, 2))

X_ngram = vectorizer_ngram.fit_transform(corpus)

tfidf_df_ngram = pd.DataFrame(
    X_ngram.toarray(),
    columns=vectorizer_ngram.get_feature_names_out()
)

print("TF-IDF with Unigrams + Bigrams:\n")
tfidf_df_ngram
print("""
Bag of Words:
- Counts word frequency
- No importance weighting

TF-IDF:
- Penalizes common words
- Highlights important terms
- Better for text classification
""")
