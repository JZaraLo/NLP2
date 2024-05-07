import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from nltk.tokenize import sent_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import spacy
import string
import numpy as np
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--json_file', type=str, help='Path to the JSON file')
args = parser.parse_args()

# Open the JSON file in reading mode
with open(args.json_file, 'r') as file:
    # Load the contents of the file into a Python object to be able to work with it
    data = json.load(file)

# Now we display sme data
print(data[:10])

# We initialize the spaCy model to perform the analysis, we use the Spanish library
nlp = spacy.load('es_core_news_sm')

# We extract the diagnosis text
text = data[0]['Text']
# We tokenize the extracted text
tokens = word_tokenize(text, language='spanish')
tokens =  [word for word in tokens if not any(char.isdigit() for char in word)]

# We clean the empty words
stop_words = set(stopwords.words('spanish'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# We lematize the words
doc = nlp(' '.join(filtered_tokens))
lemmas = [token.lemma_ for token in doc]

# We compute the TF-IDF ()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([' '.join(lemmas)])

# We obtain the names of the word's cathegories
feature_names = vectorizer.get_feature_names_out()

# We obtain the TF-IDF score for every word
tfidf_scores = tfidf_matrix.toarray()[0]

# We select the key words with higher scores
num_keywords = 10
keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:num_keywords]

# We show the key words more important of the diagnosis
print("Most important key Words of the diagnosis:")
for word, score in keywords:
    print(f"{word}: {score:.4f}")

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [token.lower() for token in sent1 if token.lower() not in stopwords]
    sent2 = [token.lower() for token in sent2 if token.lower() not in stopwords]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        vector1[all_words.index(w)] += 1

    for w in sent2:
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix
# We can vary the n value depending on the sentences we wnat to take to make the summary
def generate_summary(text, top_n=5):
    # We confirm that the NLTK downloads are available
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

    stop_words = stopwords.words('spanish')
    summarize_text = []

    # We tokenize the text in sentences
    sentences = sent_tokenize(text)

    # We build the similarity matrix between sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # We print the similarity matrix between sentences
    print("Similarity Matrix between sentences:")
    print(sentence_similarity_matrix)

    # We use the PageRank to take the most relevant sentences for the summary
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # We sort the sentences by their punctuationon PageRank
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # We select the more relavant sentences for the summary
    for i in range(top_n):
        summarize_text.append(ranked_sentences[i][1])

    # We combine the sentences in the whole summary
    summary = ' '.join(summarize_text)

    # We print the selected sentences for the summary to be able to take a look to them
    print("\nSentences selected for the summary:")
    for i, sentence in enumerate(summarize_text):
        print(f"Oración {i + 1}: {sentence}")

    return summary

# We print an example to see if it works properly
text = data[1]['Text']
summary = generate_summary(text, top_n=5)
print("\nGenerated Summary:")
print(summary)

print("\n Original Text: ",text)
