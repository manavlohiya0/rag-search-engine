from sentence_transformers import SentenceTransformer

class SemanticSearch():
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

def verify_model():
    ss = SemanticSearch()
    
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

    # return model.encode(text)

def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError("Error: Both vectors are not of the same size")
    result=[]
    for i in range(len(vec1)):
        result.append(vec1[i] + vec2[i])
    return result
    pass

def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError("Error: Both vectors are not of the same size")
    result=[]
    for i in range(len(vec1)):
        result.append(vec1[i] - vec2[i])
    return result
    pass
