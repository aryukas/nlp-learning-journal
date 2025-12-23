import numpy as np
import nltk

from nltk.tokenize import word_tokenize

nltk.download('punkt')
sentences = [
    "I love natural language processing",
    "NLP is amazing and powerful",
    "I love learning new NLP concepts",
    "Word embeddings capture semantic meaning"
]

# Cell Tokenization
tokenized_sentences = [
    word_tokenize(sentence.lower()) for sentence in sentences
]

tokenized_sentences

# Vocab 
vocab = {}
for sentence in tokenized_sentences:
    for word in sentence:
        if word not in vocab:
            vocab[word] = len(vocab)

vocab

window_size = 2
vocab_size = len(vocab)

co_occurrence = np.zeros((vocab_size, vocab_size))

for sentence in tokenized_sentences:
    for i, word in enumerate(sentence):
        word_idx = vocab[word]
        
        for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
            if i != j:
                context_word = sentence[j]
                context_idx = vocab[context_word]
                co_occurrence[word_idx][context_idx] += 1

co_occurrence
embedding_dim = 50

W = np.random.rand(vocab_size, embedding_dim)

learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    for i in range(vocab_size):
        for j in range(vocab_size):
            if co_occurrence[i][j] > 0:
                error = np.dot(W[i], W[j]) - np.log(co_occurrence[i][j])
                W[i] -= learning_rate * error * W[j]
                W[j] -= learning_rate * error * W[i]

word = "nlp"
word_vector = W[vocab[word]]

word_vector

from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

similarity = cosine_similarity(
    W[vocab["nlp"]],
    W[vocab["language"]]
)

similarity

print("""
Word2Vec:
- Predictive model
- Learns local context

GloVe:
- Count-based model
- Learns global statistics
- Better at capturing linear relationships
""")
