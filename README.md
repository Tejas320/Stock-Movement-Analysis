#  Stock Movement Analysis Based on Social Media Sentiment

## Table of Contents
1. [Objective](#objective)
2. [Task Requirements](#task-requirements)
3. [Features](#features)
4. [Project Workflow](#project-workflow)

## Objective 
Develop a machine learning model that predicts stock movements by scraping data from social media platforms like Twitter, Reddit, or Telegram. The model should extract insights from user-generated content, such as stock discussions, predictions, or sentiment analysis, and accurately forecast stock price trends.
## Task Requirements
### 1. Data Scraping: 
○ Select one platform: Twitter, Reddit, or Telegram.
○ Scrape relevant data from specific handles, subreddits, or Telegram channels focused on stock market discussions and predictions.
○ Clean and preprocess the data, ensuring it's ready for model input (e.g., removing noise, handling missing data).
### 2. Data Analysis (Optional):
○ Perform sentiment analysis or topic modeling on the scraped data.
○ Extract key features such as sentiment polarity, frequency of mentions, or any other indicators relevant to stock movements.
### 3. Prediction Model:
○ Build a machine learning model that takes the processed data as input and predicts stock movements.
○ Test the model's accuracy on historical data or known stock trends.
○ Provide a detailed evaluation of the model's performance, including metrics like accuracy, precision, recall, and any improvements that can be made.
### 4. Technical Skills Required:
○ Proficiency in Python, with experience in web scraping (using libraries such as BeautifulSoup, Scrapy, or Selenium).
○ Knowledge of Natural Language Processing (NLP) techniques for sentiment analysis and text mining.
○ Experience in building and evaluating machine learning models using libraries such as scikit-learn, TensorFlow, or PyTorch.
## Features
### 1. Data Scraping
- Uses Telethon to scrape 50k stock-related messages from Telegram channel `Stock Phoenix` which provides stock market tips and tricks.
- Extracts message text, date, message ID, views, forwards, reactions and other relevant metadata.
  
### 2. Data Preprocessing
- Noise removal: Removes URLs, emojis, special characters, and non-alphabetic content. Applied WordNetLemmatizer, removed stopwords, changed text into lowercase.
- Sentiment labeling: Sentiment of each message is classified as Positive, Negative, or Neutral.
- Handling missing data: Drops rows with NaN or empty messages.
- `Reactions` column is dropped as it contains only NaN.
  
### 3. Sentiment Analysis & Feature Engineering
- Sentiment polarity is calculated using SentimentIntensityAnalyzer. Messages having sentiment score greater than 0.05 are grouped as `Positive`, score less than -0.05 are grouped as `Negative`, and others `Neutral`. 
- Word frequency, sentiment category and sentiment scores are used as features.
- WordCloud visualizations provide insight into most frequently mentioned terms.
  
### 4. Machine Learning Models
- RandomForestClassifier for robust classification.
- Multinomial Naive Bayes for text classification.
- Support Vector Classifier (SVC) works well with high-dim features.
- Logistic Regression.
  
### 5. Evaluation
Metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
Feature Importance: Identifies the most important features affecting the stock movement.
