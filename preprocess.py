import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text: str) -> str:
    """Clean, tokenize, remove stopwords, and stem the input text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)   # remove punctuation/numbers
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)
