import os

from similarity_engine import check_against_corpus

def load_documents(folder_path: str) -> dict:
    docs = {}
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), 'r', encoding='utf-8') as f:
                docs[fname] = f.read()
    return docs

if __name__ == "__main__":
    reference_docs = load_documents("reference_corpus")
    with open("student_submission.txt", "r", encoding="utf-8") as f:
        target_text = f.read()

    report = check_against_corpus(target_text, reference_docs, threshold=0.35)

    print("Plagiarism Detection Report")
    print("-" * 40)
    for entry in report:
        pct = entry["similarity"] * 100
        print(f"{entry['source']:<25} {pct:6.2f}%   {entry['verdict']}")
