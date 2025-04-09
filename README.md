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

## Step 1: Data Preprocessing
In this step, we cleaned the dataset by removing null values, converting text to lowercase, and removing irrelevant columns like `ProductId` and `ProfileName`.

### Data Cleaning Code:
| df = df[['ProductId', 'UserId', 'Score', 'Time', 'Summary', 'Text', 
         'HelpfulnessNumerator', 'HelpfulnessDenominator']]         |

