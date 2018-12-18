from re import sub
from pickle import load

def pickle_load(filename):
    """ Read a pickled object representation and return the object.
    """
    with open(filename, 'rb') as f:
        return load(f)

def preprocess(excerpt):
    """ Return a cleaned up excerpt.
    """
    excerpt = sub('[^a-zA-Z]', ' ', excerpt)
    excerpt = excerpt.lower()
    excerpt = excerpt.split()
    excerpt = ' '.join(excerpt)
    return excerpt


def document_term_matrix(vectorizer, excerpt):
    """ Return a document-term matrix.
    """
    return vectorizer.transform([preprocess(excerpt)]).toarray()
