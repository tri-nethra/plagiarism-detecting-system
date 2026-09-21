from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess

def compute_similarity(doc_a: str, doc_b: str) -> float:
    """Return cosine similarity (0-1) between two documents."""
    corpus = [preprocess(doc_a), preprocess(doc_b)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score, 4)

def check_against_corpus(target_text: str, reference_docs: dict, threshold: float = 0.35):
    """
    Compare target_text against every document in reference_docs
    (a dict of {filename: text}) and return a sorted similarity report.
    """
    results = []
    for fname, ref_text in reference_docs.items():
        score = compute_similarity(target_text, ref_text)
        verdict = "PLAGIARISED" if score >= threshold else "ORIGINAL"
        results.append({"source": fname, "similarity": score, "verdict": verdict})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results
