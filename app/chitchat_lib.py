from re import sub

def preprocess(excerpt):
    """ Return a cleaned up excerpt without stopwords.
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