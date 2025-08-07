from collections import Counter 
from collections import defaultdict
import math

class BoW: 
    def __init__(self):
        self.vocab_freq= Counter()
    def bow(self, text): 
        print(text)
        self.vocab_freq.update(text.lower().split())
        return self.vocab_freq

class TFIDF: 
    def __init__(self):
        self.tf=defaultdict(int)
        self.idf = defaultdict(int)

    def compute_tf(self,doc):
        for word in doc.split():
            self.tf[word] += 1
        total_terms = len(doc.split())
        for word in self.tf:
            self.tf[word] /= total_terms
        return self.tf 
    
    def compute_idf(self,docs):
        N = len(docs)
        for doc in docs:
            words = set(doc.split())
            for word in words:
                self.idf[word] += 1
        for word in self.idf:
            self.idf[word] = math.log(N / (1 + self.idf[word]))  # smooth with +1
        return self.idf
    
    def compute_tfidf(self, corpus):
        tfidf_matrix = []
        idf = self.compute_idf(corpus)
        for doc in corpus:
            tf = self.compute_tf(doc)
            tfidf = {word: tf[word] * idf[word] for word in tf}
            tfidf_matrix.append(tfidf)
        return tfidf_matrix


        
bow = BoW()
text = " I love ice-cream. I also love paris"
print(bow.bow(text))

tfidf = TFIDF()
text = [" I love ice-cream.", " I also love paris", "Paris is my dream cream city"]
print(tfidf.compute_tfidf(text))