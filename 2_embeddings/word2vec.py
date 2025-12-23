import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Download tokenizer (run once)
nltk.download('punkt')

sentences = [
    "I love natural language processing",
    "NLP is amazing and powerful",
    "I love learning new NLP concepts",
    "Word embeddings capture semantic meaning"
]

# Tokenize
tokenized_sentences = [
    word_tokenize(sentence.lower()) for sentence in sentences
]

# Train Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Vector for a word
print("Vector for 'nlp':")
print(model.wv['nlp'])

# Similar words
print("\nSimilar words to 'nlp':")
print(model.wv.most_similar('nlp'))

# Similarity score
print("\nSimilarity between 'nlp' and 'language':")
print(model.wv.similarity('nlp', 'language'))

# Save model
model.save("word2vec.model")
