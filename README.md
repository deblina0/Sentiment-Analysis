# Sentiment-Analysis

##**Problem Statememt**


**Sentiment Analysis of Product Reviews:**

- **Description:** Analyse customer reviews to determine sentiment using natural language processing (NLP) techniques. This helps in understanding customer opinions and improving product features.

- **Why:** Sentiment analysis provides insights into customer satisfaction
and areas needing improvement.

- **Tasks:**

    ▪ Collect product review data.

    ▪ Example datasets Click Here

    ▪ Preprocess text data (tokenization, stop word removal).

    ▪ Apply NLP techniques (e.g., sentiment 
    analysis using NLTK or spaCy).

    ▪ Visualize and interpret results

from google.colab import files
uploaded = files.upload()

import pandas as pd

sn = pd.read_csv("sentiment_analysis.csv")

sn

# import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Preprocess function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercasing
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Stop word removal
    return ' '.join(tokens)

# Apply preprocessing
sn['cleaned_text'] = sn['text'].apply(preprocess_text)

from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to get sentiment
def get_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis
sn['predicted_sentiment'] = sn['cleaned_text'].apply(get_sentiment)

import matplotlib.pyplot as plt
import seaborn as sns

# Count sentiments
sentiment_counts = sn['predicted_sentiment'].value_counts()

# Plotting
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
