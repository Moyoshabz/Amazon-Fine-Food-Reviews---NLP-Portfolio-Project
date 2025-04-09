# Amazon Fine Food Reviews

This project applies Natural Language Processing (NLP) techniques to extract insights from food product reviews posted on Amazon. It's built using Python, Pandas, NLTK/SpaCy, and Matplotlib, and is designed as a showcase portfolio project.

---

## Project Goals:

- Clean and preprocess real-world customer review data
- Tokenize text data for further analysis
- Perform exploratory data analysis (EDA)
- Visualize word frequencies with WordCloud
- Conduct sentiment analysis (coming soon!)
- Apply machine learning models for sentiment classification (optional)
- Share business insights and recommendations

**Source**: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  
**Records**: 568,000+ customer reviews  
**Key Features**:
- `Score` (rating 1–5)
- `Text` (the actual review)
- Additional metadata (not all used)

---

## 📊 Methods & Technologies

| Task               | Tools & Libraries Used                     |
|--------------------|--------------------------------------------|
| Data Loading       | pandas                                     |
| Text Preprocessing | nltk                                       |
| Visualization      | matplotlib, seaborn, wordcloud             |
| Sentiment Analysis | textblob or vader (coming soon)            |
| Classification     | scikit-learn (optional for future section) |

---

## Step 1: Data Preprocessing and Feature Engineering
In this step, we cleaned the dataset by removing null values, converting text to lowercase, and removing irrelevant columns like `ProductId` and `ProfileName`. 

### Data Cleaning Code:
df.dropna(inplace=True)

df = df[['ProductId', 'UserId', 'Score', 'Time', 'Summary', 'Text', 
         'HelpfulnessNumerator', 'HelpfulnessDenominator']]

df['Text'] = df['Text'].str.lower()
df['Summary'] = df['Summary'].str.lower()

Also we coverted the current UNIX timestamp in our dataset to datetime

df['ReviewTime'] = pd.to_datetime(df['Time'], unit='s')

### Creating Additional Features:
Additionally, we created new features that could help in understanding the reviews and their characteristics

df['review_length'] = df['Text'].apply(lambda x: len(str(x).split()))
df['summary_length'] = df['Summary'].apply(lambda x: len(str(x).split()))
df['helpfulness_ratio'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'].replace(0, 1)
df['sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))

---

## Step 2: Tokenization
We tokenized the review text using NLTK and cleaned the tokens by removing stopwords.

import nltk
nltk.download('punkt_tab')
df['tokens'] = df['Text'].apply(word_tokenize)


from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df['tokens_clean'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

df[['tokens', 'tokens_clean']].head()

After the tokenization had been implemented, we visulized the most frequent words in our review and noticed that symbols are chraters such as 'I', 'br' were prevalent. we took them out using the following code .

import string
punctuation = set(string.punctuation)
df['tokens_clean'] = df['tokens'].apply(lambda x: [word for word in x if word.isalpha() and word not in stop_words])
df[['tokens', 'tokens_clean']].head()

--- 

## Step 3: Tokenization
We classified each review as positive, negative, or neutral based on the review score using a simple rule-based approach.

df['sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))

df['sentiment']

---

## Step 4: Data Visualizations
In this step, we created various visualizations to explore the dataset and uncover patterns related to sentiment, review characteristics, and overall trends

1. Sentiment Distribution Based on Score : We visualized how the sentiment distribution varies across different review scores. This helps us understand if higher scores correlate with more positive sentiments or if negative sentiments can still appear in high-rated reviews

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data=df, x='sentiment', order=['positive', 'neutral', 'negative'])
plt.title("Sentiment Distribution Based on Score")
plt.show()






- Distribution of Review Lengths
- Helpfulness Ratio by Sentiment
- Time Series: Reviews Over Time
- Word Cloud for Full Review Text and Summary
- Top 20 Most Frequent Words in Reviews
- 







