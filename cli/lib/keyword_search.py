from typing import TypedDict
import json
import string
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import os
import pickle
import math
from .utils import BM25_K1, BM25_B 

class InvertedIndex():

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        
        self.index_path = './cache/index.pkl'
        self.docmap_path = './cache/docmap.pkl'
        self.term_frequencies_path = './cache/term_frequencies.pkl'
        self.doc_lengths_path = './cache/doc_lengths.pkl'

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m['id']
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        for token in tokens:
            self.term_frequencies[token][doc_id] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0: return 0.0
        return sum(self.doc_lengths.values())/len(self.doc_lengths)

    def get_idf(self, term: str) -> float:
        term_match_doc_count = len(self.term_frequencies[term])
        total_doc_count = len(self.docmap)
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[term][doc_id]

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.term_frequencies[term])
        return math.log(1+(total_doc_count - term_match_doc_count + 0.5)/(term_match_doc_count + 0.5))

    def get_bm25_tf(self, doc_id:int, term:str, k1:float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length) 
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int) -> str:
        query_tokens = tokenize_text(query)
        scores: dict[int, float] = {}
        for token in query_tokens:
            for doc_id in self.index[token]:
                total_doc_score = scores.get(doc_id,0)
                total_doc_score += self.bm25(doc_id, token)    
                scores[doc_id] = total_doc_score
        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return scores[:limit]

    def save(self) -> None:
        os.makedirs('./cache/', exist_ok = True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f: 
            pickle.dump(self.term_frequencies, f)
        with open (self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    term = tokenize_term(term)
    return idx.get_bm25_idf(term)

def bm25_search_command(query: str, limit: int = 5) -> dict:
    idx = InvertedIndex()
    idx.load()
    scores = idx.bm25_search(query, limit)
    results = []
    for id, score in scores:
        results.append({"id":id,"title":idx.docmap[id]['title'],"score":score})
    return results

def bm25_tf_command(doc_id: int,term: str, k1:float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    term = tokenize_term(term)
    return idx.get_bm25_tf(doc_id, term, k1, b)

def idf_command(term:str) -> float:
    idx = InvertedIndex()
    idx.load()
    term = tokenize_term(term)
    return idx.get_idf(term)

def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    term = tokenize_term(term)
    return idx.get_tf_idf(doc_id, term)

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    term = tokenize_term(term)
    return idx.get_tf(doc_id, term)

class Movie(TypedDict):
    id: int
    title: str
    description: str

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def load_stopwords() -> list[str]: 
    with open("/home/manavl/boot.dev/rag-search-engine/data/stopwords.txt", "r") as file:
        return [preprocess_text(word) for word in file.read().splitlines()]

STOPWORDS = load_stopwords()

def get_movies_from_query(idx: InvertedIndex, query_tokens: list[str]) -> list[Movie]:
    results=[]
    for token in query_tokens: 
        docs = idx.get_documents(token)
        for doc_id in docs:
            results.append(idx.docmap[doc_id])
            if len(results) >= 5:
                return results
    return results

def search_command(query: str) -> list[Movie]:
    idx = InvertedIndex()
    idx.load()
    
    query_tokens = tokenize_text(query)
    return get_movies_from_query(idx, query_tokens)

def load_movies() -> list[Movie]:
    with open('/home/manavl/boot.dev/rag-search-engine/data/movies.json','r') as f:
        movies = json.load(f)
    return movies['movies']

def tokenize_term(text: str) -> str:
    token = tokenize_text(text)
    if len(token) == 1:
        return token[0]
    else: 
        raise ValueError("More than one token!")

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
