# Chit-chat Classifier

Requirements: Python 3.6+, Scikit-learn, Scipy, Numpy

Navigate to the `app` directory and install dependencies:

```bash
$ pip install -r requirements.txt
```

Run the application:

```bash
$ python main.py
```

In your web browser, enter the following address:

```
http://localhost:8080
```

POST-request example using [cURL](https://curl.haxx.se/):

```bash
$ curl http://localhost:8080/classify -d "data=Let's have lunch and catch up next week"
```

Output:

```json
{
    "text": "Let's have lunch and catch up next week", 
    "classification": {
        "class_name": "Chit-chat", 
        "probability": 0.8134498791362933
    }
}
```

In your terminal window, press **Ctrl+C** to exit the web server.
