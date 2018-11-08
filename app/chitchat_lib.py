from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from re import sub

def preprocess(excerpt):
    """ Returns a cleaned up excerpt without stopwords
    """
    excerpt = sub('[^a-zA-Z]', ' ', excerpt)
    excerpt = excerpt.lower()
    excerpt = excerpt.split()
    excerpt = [word for word in excerpt if not word in set(stopwords.words('english'))]
    excerpt = ' '.join(excerpt)
    return excerpt


def document_term_matrix(excerpt):
    """ Creates and returns a document-term matrix
    """
    tfidf = joblib.load('model/tfidf.vectorizer')
    vec = CountVectorizer(vocabulary=tfidf.vocabulary_)
    return vec.transform([preprocess(excerpt)]).toarray()