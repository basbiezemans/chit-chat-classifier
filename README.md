# Chit-chat Classifier

Requirements: Python 3.6+, Sklearn, Nltk

```bash
$ pip install -r requirements.txt
```

You can run the classifier script with an excerpt as argument:

```bash
$ python chitchat_classifier.py "Let's have lunch and catch up next week"
```

Output:

```json
{
    "text": "Let's have lunch and catch up next week", 
    "classification": {
        "class_name": "Chit-chat", 
        "probability": 0.6174352739594865
    }
}
```

