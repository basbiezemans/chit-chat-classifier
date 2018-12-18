from flask import Flask, request, jsonify
from sklearn.externals import joblib
from chitchat_classifier import ChitChatClassifier

app = Flask(__name__)

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
    try:
        mdl = joblib.load('model/chitchat.model')
        vec = joblib.load('model/tfidf.vectorizer')
        clf = ChitChatClassifier(mdl, vec)
        app.run(debug=True, host='127.0.0.1', port=8080)
    except Exception as e:
        print(str(e))
