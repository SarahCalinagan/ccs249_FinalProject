from flask import Flask, render_template, request, jsonify
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
# Load NLP models
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

@app.route('/')
def index():
    # Render an HTML form for user input
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # NLP Processing
    doc = nlp(text)
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    sentiment = sia.polarity_scores(text)
    
    # Format sentiment scores
    formatted_sentiment = {
        'Negative': "{:.2f}".format(sentiment['neg']),
        'Neutral': "{:.2f}".format(sentiment['neu']),
        'Positive': "{:.2f}".format(sentiment['pos']),
        'Compound': "{:.2f}".format(sentiment['compound'])
    }
    # Convert formatted sentiment scores to strings with newlines
    formatted_sentiment_str = "\n".join([f"{key}: {value}" for key, value in formatted_sentiment.items()])
    

    # Sentiment prediction logic
    compound = sentiment['compound']
    if compound > 0.1:
        image_url = "/static/happy.png"
        prediction = "Positive"
    elif compound < -0.1:
        image_url = "/static/nega.png"
        prediction = "Negative"
    else:
        image_url = "/static/neutral.png"
        prediction = "Neutral"
    
    tokens_length = len(tokens)
    pos_tags_length = len(pos_tags)
    
    # Render a results page with the sentiment information
    return render_template('result.html', tokens_length=tokens_length, pos_tags_length=pos_tags_length, tokens=tokens, pos_tags=pos_tags, sentiment=formatted_sentiment_str, prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
