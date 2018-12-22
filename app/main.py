from flask import Flask, request, jsonify
from sklearn.externals import joblib
from chitchat_classifier import ChitChatClassifier

app = Flask(__name__)
clf = None

@app.before_first_request
def load_model():
    """ Recreate the model from file.
    """
    global clf
    mdl = joblib.load('model/chitchat.model')
    vec = joblib.load('model/tfidf.vectorizer')
    clf = ChitChatClassifier(mdl, vec)

@app.route('/')
def test():
    """ Return a test message.
    """
    return 'The web server is up and running.'

@app.route('/classify', methods=['POST'])
def classify():
    """ Return a JSON response with the prediction for an excerpt.
    """
    excerpt = request.form.get('data')
    if excerpt is None or not excerpt:
        return jsonify(message='Bad request'), 400
    return jsonify(clf.predict(excerpt))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
