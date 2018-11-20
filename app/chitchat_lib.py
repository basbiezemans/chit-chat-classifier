from sklearn.externals import joblib
from re import sub

def preprocess(excerpt):
    """ Returns a cleaned up excerpt
    """
    excerpt = sub('[^a-zA-Z]', ' ', excerpt)
    excerpt = excerpt.lower()
    excerpt = excerpt.split()
    excerpt = ' '.join(excerpt)
    return excerpt


def document_term_matrix(excerpt):
    """ Creates and returns a document-term matrix
    """
    vec = joblib.load('model/tfidf.vectorizer')
    return vec.transform([preprocess(excerpt)]).toarray()
