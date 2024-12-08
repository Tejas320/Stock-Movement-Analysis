#  Stock Movement Analysis Based on Social Media Sentiment

## Table of Contents
1. [Objective](#objective)
2. [Task Requirements](#task-requirements)
3. [Features](#features)
4. [How to Run the Project](#how-to-run-the-project)
5. [Key Results](#key-results)
6. [Feature Importance](#feature-importance)
7. [Technologies Used](#technologies-used)
8. [Future Improvements](#future-improvements)

##  📚 Objective 
Develop a machine learning model that predicts stock movements by scraping data from social media platforms like Twitter, Reddit, or Telegram. The model should extract insights from user-generated content, such as stock discussions, predictions, or sentiment analysis, and accurately forecast stock price trends.
## Task Requirements
### 1. Data Scraping: 
- Select one platform: Twitter, Reddit, or Telegram.
- Scrape relevant data from specific handles, subreddits, or Telegram channels focused on stock market discussions and predictions.
- Clean and preprocess the data, ensuring it's ready for model input (e.g., removing noise, handling missing data).
### 2. Data Analysis (Optional):
- Perform sentiment analysis or topic modeling on the scraped data.
- Extract key features such as sentiment polarity, frequency of mentions, or any other indicators relevant to stock movements.
### 3. Prediction Model:
- Build a machine learning model that takes the processed data as input and predicts stock movements.
- Test the model's accuracy on historical data or known stock trends.
- Provide a detailed evaluation of the model's performance, including metrics like accuracy, precision, recall, and any improvements that can be made.
### 4. Technical Skills Required:
- Proficiency in Python, with experience in web scraping (using libraries such as BeautifulSoup, Scrapy, or Selenium).
- Knowledge of Natural Language Processing (NLP) techniques for sentiment analysis and text mining.
- Experience in building and evaluating machine learning models using libraries such as scikit-learn, TensorFlow, or PyTorch.
## 🔧 Features
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
## 📂 How to Run the Project
### 1️. Clone the Repository
```bash
git clone https://github.com/Tejas320/Stock-Movement-Analysis.git
cd stock-movement-analysis
```
### 2. Set up Telegram API
- Visit my.telegram.org.
- Get your API ID and API Hash.
- Add the credentials to `scraping.py`
  ```bash
  api_id = 'YOUR_API_ID'
  api_hash = 'YOUR_API_HASH'
  ```
### 3. Run Data Scraper
Run the data scraper `scraping.py` to collect stock-related messages from a Telegram channel or you can direct use the dataset provided.
```bash
python scraping.py
```
## 🔍 Key Results
- Random Forest achieved an accuracy of `93%` on the test set.
- Multinomial Naive Bayes had an accuracy of `76%`, but was faster to train.
- Support Vector Classifier outperformed other models with an accuracy of `95%`.
- Logistic Regression performed similar to Random Forest with accuracy of `93%`.
#### Classification report of SVC
  ![image](https://github.com/user-attachments/assets/ae797c60-f399-49de-a461-a3458128b382)
#### Confusion Matrix of SVC model
![image](https://github.com/user-attachments/assets/60fb1436-6e59-4b1c-adcb-47491952c14d)
## 📊 Feature Importance
The most important features for stock movement predictions were:
- Sentiment polarity score
- Word frequencies of terms like "nifty", "bearish", "stop", "profit"
#### Sentiment Distribution
![image](https://github.com/user-attachments/assets/048bfdff-81d6-4094-9efb-cc62086daa18)
#### Top 20 Important Features
![image](https://github.com/user-attachments/assets/aefe2b15-7a65-451c-a4e2-6c7efcdac655)
#### Wordcloud
![image](https://github.com/user-attachments/assets/545e6273-7e89-46bb-8eef-fc3a61419ce2)
## ⚙️ Technologies Used
- Programming Language: Python
- Data Scraping: Telethon
- Data Analysis & Visualization: Pandas, Seaborn, Matplotlib, WordCloud
- Machine Learning: Sklearn, RandomForest, Naive Bayes, SVC, LogisticRegression
## 🚀 Future Improvements
- Add Sentiment Analysis from more channels of Telegram.
- Use transformer-based models like BERT for better text classification.
- Implement LSTM or GRU models for better time-series forecasting.

