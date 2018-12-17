from chitchat_lib import document_term_matrix

class ChitChatClassifier:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vec = vectorizer
        self.labels = ('Not chit-chat', 'Chit-chat')

    def predict(self, excerpt):
        x = document_term_matrix(self.vec, excerpt)
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
