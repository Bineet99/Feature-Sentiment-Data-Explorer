import pandas as pd
from keybert import KeyBERT
from textblob import TextBlob
from collections import defaultdict
import matplotlib.pyplot as plt

# Product reviews from CSV file
df = pd.read_csv("sample_product_reviews_1000.csv")

# Initialize KeyBERT
kw_model = KeyBERT()

# Extract top 3 keywords from each review
def extract_top3_keywords(text):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=3
    )
    return [kw[0] for kw in keywords]

df['features'] = df['review_text'].apply(extract_top3_keywords)

# Analyze sentiment of each review
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['review_text'].apply(get_sentiment)

# Count sentiment per feature
feature_sentiment = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0})

for _, row in df.iterrows():
    sentiment = row['sentiment']
    for feature in row['features']:
        feature_sentiment[feature][sentiment] += 1
        feature_sentiment[feature]['total'] += 1

# Create summary table
summary = []
for feature, stats in feature_sentiment.items():
    total = stats['total']
    summary.append({
        'Feature': feature,
        'Total Mentions': total,
        'Positive': stats['positive'],
        'Negative': stats['negative'],
        'Neutral': stats['neutral'],
        '% Positive': round(100 * stats['positive'] / total, 2),
        '% Negative': round(100 * stats['negative'] / total, 2)
    })

summary_df = pd.DataFrame(summary).sort_values(by='Total Mentions', ascending=False)

# Plot sentiment distribution by feature
plt.figure(figsize=(12, 6))
plt.bar(summary_df['Feature'], summary_df['Positive'], color='green', label='Positive')
plt.bar(summary_df['Feature'], summary_df['Negative'], bottom=summary_df['Positive'], color='red', label='Negative')
plt.bar(summary_df['Feature'], summary_df['Neutral'],
        bottom=summary_df['Positive'] + summary_df['Negative'], color='gray', label='Neutral')

plt.title("Sentiment Distribution by Extracted Features (KeyBERT, 1000 Reviews)")
plt.xlabel("Feature")
plt.ylabel("Number of Mentions")
plt.xticks(rotation=45, ha='right')
plt.show()

# Print summary table
print(summary_df.reset_index(drop=True))
