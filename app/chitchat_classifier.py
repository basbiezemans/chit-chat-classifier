from sklearn.externals import joblib
from chitchat_lib import document_term_matrix
from sys import argv
import json

class ChitChatClassifier:
    def __init__(self):
        self.model = joblib.load('model/chitchat.model')
        self.labels = ('Not chit-chat', 'Chit-chat')

    def predict(self, excerpt):
        x = document_term_matrix(excerpt)
        p = self.model.predict_proba(x)[0]
        d = dict(zip(self.model.classes_, p))
        key = max(d, key=d.get)
        return {
            'text': excerpt,
            'classification': {
                'class_name': self.labels[key],
                'probability': d[key]
            }
        }

if __name__ == '__main__':
    if len(argv) > 1:
        clf = ChitChatClassifier()
        print(json.dumps(clf.predict(argv[1])))
    else:
        print('Error: argument missing.')
        print('Usage: python chitchat_classifier.py <excerpt>')