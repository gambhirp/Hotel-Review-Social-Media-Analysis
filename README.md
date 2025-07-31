# Hotel Review Analysis – Social Media Project
### Project Overview
This project involves the analysis of over 515,000 hotel reviews sourced from Booking.com. It uses natural language processing, sentiment analysis, and machine learning to derive actionable insights. The aim is to benchmark hotel performance and identify areas for enhancing the guest experience.

### Objectives
The main goals of this project are to classify hotel reviews as either positive or negative, uncover recurring themes using topic modeling techniques such as Latent Dirichlet Allocation, benchmark hotels based on sentiment scores, prices, and star ratings, and build an interactive web application using Flask to display insights in a user-friendly format.

### Methodology
The dataset was prepared by merging positive and negative reviews into one combined dataset. The text was cleaned by removing unnecessary characters, tokenized, and lemmatized. Sentiment labels were assigned based on review scores, where reviews rated seven or above were labeled as positive, and those rated below seven were labeled as negative. The dataset was balanced with 86,851 positive and 86,851 negative reviews to ensure model accuracy.

TF-IDF vectorization was applied using the top 5,000 terms to create a high-dimensional sparse matrix suitable for model input. Two machine learning models were tested: Naive Bayes and Logistic Regression. The Logistic Regression model was selected as the final classifier due to its performance. LIME (Local Interpretable Model-agnostic Explanations) was used to provide transparency and interpretability for the predictions.

Separate topic models were built for positive and negative reviews using Latent Dirichlet Allocation. Five topics were extracted from each category. Common negative themes included noise issues, poor Wi-Fi connectivity, and bathroom-related complaints. On the positive side, guests frequently praised the staff, room comfort, and cleanliness.

### Sentiment by Country
Sentiment analysis by country revealed that Austria had the highest average review scores. The United Kingdom generated the most reviews, though the sentiment expressed was generally lower. The Netherlands exhibited strong sentiment scores relative to the average numerical ratings, indicating a more favorable perception.

### Key Insights
Frequent travelers tended to provide more balanced and objective reviews. The specific sentiment-related vocabulary varied depending on the type of traveler and their nationality. Regional differences were observed in both the volume and nature of reviews.

### Hotel Benchmarking
Hotels were grouped into three performance tiers based on a combination of sentiment scores and review features. GPT-based text summarization was used to highlight the most frequently mentioned positive and negative aspects of each tier. Common issues across underperforming hotels included noise disturbances, inconsistent cleanliness, and service delays.

### Tools and Technologies
The analysis was conducted using Python and libraries such as pandas, scikit-learn, NLTK, Gensim, and WordCloud. LIME was used for model explainability, while Flask was used to develop the web application. GPT API was employed for automated summary generation. Data visualizations were created using Matplotlib, including radar charts to compare hotel attributes.

### Deliverables
The final deliverables include a working sentiment classifier with interpretability features, topic models that summarize guest feedback, word clouds to visualize common themes, and an interactive web dashboard alongside a command-line interface tool for hotel comparison.

### Conclusion
This project demonstrates how natural language processing and machine learning can be used to transform large volumes of unstructured hotel review data into meaningful insights. These insights can help hospitality businesses make informed, data-driven decisions that are aligned with guest preferences and expectations.
