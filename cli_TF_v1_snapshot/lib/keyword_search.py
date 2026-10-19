from typing import TypedDict
import json
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
import os
import pickle

class InvertedIndex():

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = './cache/index.pkl'
        self.docmap_path = './cache/docmap.pkl'

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

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def save(self) -> None:
        os.makedirs('./cache/', exist_ok = True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

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
            if len(results) >= 20:
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
