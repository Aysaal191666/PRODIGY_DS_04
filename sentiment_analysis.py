import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re

# Load dataset
file_path = "twitter_training.csv"  
df = pd.read_csv(file_path, encoding="utf-8", header=None)

# Rename columns based on dataset structure
df.columns = ["index", "entity", "sentiment", "text"]

# Keep only necessary columns
df = df[['text', 'sentiment']]

# Remove null values
df.dropna(inplace=True)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Filter for only positive and negative sentiments
df = df[df['sentiment'].isin(["Positive", "Negative"])]

# Split data into positive and negative sentiments
positive_text = ' '.join(df[df['sentiment'] == 'Positive']['clean_text'])
negative_text = ' '.join(df[df['sentiment'] == 'Negative']['clean_text'])

# Define stopwords
stopwords = set(STOPWORDS)

# Generate WordClouds
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

wordcloud_positive = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(positive_text)
axes[0].imshow(wordcloud_positive, interpolation='bilinear')
axes[0].set_title("Positive Sentiment Word Cloud", fontsize=14)
axes[0].axis('off')

wordcloud_negative = WordCloud(width=800, height=400, background_color='black', stopwords=stopwords, colormap='Reds').generate(negative_text)
axes[1].imshow(wordcloud_negative, interpolation='bilinear')
axes[1].set_title("Negative Sentiment Word Cloud", fontsize=14)
axes[1].axis('off')

# Save and show
plt.tight_layout()
plt.savefig("sentiment_wordcloud.png")
plt.show()
