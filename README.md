# 🏨 Hotel Review Analysis – Social Media Project

## 📌 Project Overview

This project analyzes 515,000+ Booking.com hotel reviews using NLP, sentiment analysis, and machine learning to extract insights, benchmark hotel performance, and improve guest experience.


## 🎯 Objectives

- Classify reviews as **positive** or **negative**
- Discover key **themes** using topic modeling (LDA)
- **Benchmark hotels** based on sentiment, price, and star ratings
- Build an **interactive Flask web app** for end-user insights


## 🧠 Methodology

### 🔄 Data Preparation
- Merged positive/negative reviews
- Cleaned text, removed placeholders, tokenized, and lemmatized
- Labeled sentiment (≥7 = Positive, <7 = Negative)
- Balanced dataset: 86,851 positive + 86,851 negative reviews

### 📊 Feature Engineering
- TF-IDF vectorization (Top 5,000 words)
- Created high-dimensional matrix for model input

### 🤖 Models Used
- **Naive Bayes**  
- **Logistic Regression** (Final model)  
- **LIME** for model explainability
  

## 📌 Topic Modeling (LDA)

- Separate models for positive and negative reviews
- Extracted 5 topics per group
- Highlighted key pain points (e.g., noise, Wi-Fi, bathrooms)
- Identified guest favorites (e.g., staff, comfort, cleanliness)


## 🌍 Sentiment by Country

- Austria showed the highest average scores
- UK had most reviews but lower sentiment
- Netherlands had strong sentiment relative to scores


## 💡 Insights

- Frequent travelers gave more balanced ratings
- Sentiment words varied based on travel type and nationality
- Review volume varied by region


## 📈 Hotel Benchmarking

- Grouped hotels into 3 tiers
- Used **GPT-based summaries** to highlight guest likes/dislikes
- Common issues: noise, cleanliness, delays


## 🛠 Tools & Technologies

- **Python** (pandas, scikit-learn, NLTK, Gensim, WordCloud)
- **LIME** (model explainability)
- **Flask** (web app)
- **GPT API** (summary generation)
- **Matplotlib / Radar Charts** (visual insights)


## 🧪 Deliverables

- Sentiment classifier with LIME explainability
- Topic models and keyword summaries
- Word clouds for visual theme analysis
- Web dashboard + CLI tool for hotel comparison


## ✅ Conclusion

This project demonstrates how NLP and machine learning can transform unstructured hotel review data into actionable business intelligence — empowering hospitality providers to make data-driven, guest-focused decisions.


## 📁 Repository Contents

- `data/` – cleaned and processed datasets  
- `notebooks/` – EDA, modeling, LDA, visualizations  
- `app/` – Flask web application  
- `cli/` – GPT-powered benchmarking CLI tool  
- `report/` – Final presentation and documentation  

