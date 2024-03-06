import numpy as np
import spacy
import json

class Vectordb:
    def __init__(self, n_dimensions=100, min_df=1):
        self.n_dimensions = n_dimensions
        self.min_df = min_df
        self.word_index = {}
        self.idf = None
        self.nlp = spacy.load("en_core_web_sm")

    def _preprocess(self, text):
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_punct]
        return tokens

    def fit(self, texts):
        term_frequency = {}
        num_docs = len(texts)

        for text in texts:
            tokens = self.preprocess(text)
            for token in set(tokens):
                term_frequency[token] = term_frequency.get(token, 0) + 1

        vocab = [word for word, freq in term_frequency.items() if freq >= self.min_df]
        vocab.sort()
        self.word_index = {word: i for i, word in enumerate(vocab)}
        
        df = np.zeros(len(vocab))
        for text in texts:
            tokens = self.preprocess(text)
            tokens = list(set(tokens)) 
            for token in tokens:
                if token in self.word_index:
                    df[self.word_index[token]] += 1
        
        self.idf = np.log(num_docs / (df + 1))

    def _transform(self, text):
        tokens = self.preprocess(text)
        vector = np.zeros(len(self.word_index))

        for token in tokens:
            if token in self.word_index:
                index = self.word_index[token]
                vector[index] += 1

        tfidf_vector = vector * self.idf
        if len(tfidf_vector) > self.n_dimensions:
            tfidf_vector = tfidf_vector[:self.n_dimensions]
        elif len(tfidf_vector) < self.n_dimensions:
            tfidf_vector = np.pad(tfidf_vector, (0, self.n_dimensions - len(tfidf_vector)), 'constant')
        return tfidf_vector

    def save_to_json(self, filename):
        data = {
            "word_index": self.word_index,
            "idf": self.idf.tolist()
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

    @classmethod
    def load_from_json(cls, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        instance = cls()
        instance.word_index = data['word_index']
        instance.idf = np.array(data['idf'])
        instance.n_dimensions = len(instance.idf)
        return instance
