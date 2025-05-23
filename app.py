from flask import Flask, render_template, request, send_file
import joblib
import lime.lime_text
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
from wordcloud import WordCloud
from werkzeug.utils import secure_filename
from openai import OpenAI
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- Load ML Models ---
class_names = ['Negative', 'Positive']
explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)

try:
    nb_model = joblib.load("models/nb_model.pkl")
    lr_model = joblib.load("models/lr_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
except Exception as e:
    print("Error loading models:", e)

# --- Load Hotel Review Dataset Once ---
df = pd.read_csv("Hotel_Reviews.csv")
df['Positive_Review'] = df['Positive_Review'].replace('No Positive', '')
df['Negative_Review'] = df['Negative_Review'].replace('No Negative', '')
df['Review'] = df['Negative_Review'].fillna('') + ' ' + df['Positive_Review'].fillna('')
df['Sentiment'] = df['Reviewer_Score'].apply(lambda x: 'Negative' if x < 7 else 'Positive')

hotel_sentiment_summary = df.groupby(['Hotel_Name', 'Sentiment']).size().unstack(fill_value=0)
hotel_sentiment_summary['Total_Reviews'] = hotel_sentiment_summary.sum(axis=1)
hotel_sentiment_summary['Positive_Percentage'] = (
    hotel_sentiment_summary['Positive'] / hotel_sentiment_summary['Total_Reviews']
) * 100
hotel_sentiment_summary['Tier'] = pd.qcut(
    hotel_sentiment_summary['Positive_Percentage'], q=3, labels=['Low', 'Mid', 'Top']
)

# --- OpenAI Setup ---
client = OpenAI(api_key="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")


def get_emoji(sentiment):
    return "😄" if sentiment == "Positive" else "😞" if sentiment == "Negative" else "😐"

def generate_radar_chart(hotel_name, hotel_reviews):
    themes = {
        'Cleanliness': ['clean', 'dirty', 'smell'],
        'Staff': ['staff', 'rude', 'friendly'],
        'Location': ['location', 'metro', 'access'],
        'Comfort': ['comfortable', 'bed', 'noise'],
        'Amenities': ['wifi', 'breakfast', 'internet']
    }

    text = " ".join(hotel_reviews['Review'].tolist()).lower()
    scores = [sum(text.count(k) for k in keywords) for keywords in themes.values()]
    max_score = max(scores) or 1
    scores = [s / max_score for s in scores]

    labels = list(themes.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, color='teal', linewidth=2)
    ax.fill(angles, scores, color='teal', alpha=0.15)
    ax.set_title(f"{hotel_name} – Guest Experience", fontsize=15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    path = os.path.join("static", f"{hotel_name}_radar.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return f"{hotel_name}_radar.png"

# --- Routes ---

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        review = request.form.get('review')
        file = request.files.get('review_file')

        if review and review.strip():
            X_input = vectorizer.transform([review])
            lr_pred = lr_model.predict(X_input)[0]
            lr_proba = lr_model.predict_proba(X_input)[0]
            lr_sentiment = class_names[lr_pred]
            lr_confidence = round(lr_proba[lr_pred] * 100, 2)

            nb_pred = nb_model.predict(X_input)[0]
            nb_proba = nb_model.predict_proba(X_input)[0]
            nb_sentiment = class_names[nb_pred]
            nb_confidence = round(nb_proba[nb_pred] * 100, 2)

            pipeline = lambda x: lr_model.predict_proba(vectorizer.transform(x))
            exp = explainer.explain_instance(review, pipeline, num_features=10)
            lime_html = exp.as_html()

            word_weights = dict(exp.as_list())
            positive_words = {w: abs(wt) for w, wt in word_weights.items() if wt > 0}
            negative_words = {w: abs(wt) for w, wt in word_weights.items() if wt < 0}

            wc_pos = WordCloud(width=400, height=250, background_color="white").generate_from_frequencies(positive_words)
            wc_neg = WordCloud(width=400, height=250, background_color="white").generate_from_frequencies(negative_words)

            pos_path = os.path.join("static", "wordcloud_positive.png")
            neg_path = os.path.join("static", "wordcloud_negative.png")
            wc_pos.to_file(pos_path)
            wc_neg.to_file(neg_path)

            return render_template(
                "result.html",
                review=review,
                lr_sentiment=lr_sentiment,
                lr_confidence=lr_confidence,
                nb_sentiment=nb_sentiment,
                nb_confidence=nb_confidence,
                lime_html=lime_html,
                word_weights=word_weights,
                pos_wc_path="wordcloud_positive.png",
                neg_wc_path="wordcloud_negative.png"
            )

        elif file and file.filename:
            filename = secure_filename(file.filename)

            if filename.endswith('.csv'):
                df_input = pd.read_csv(file)
            elif filename.endswith(('.xls', '.xlsx')):
                df_input = pd.read_excel(file)
            else:
                return "Unsupported file type. Please upload a .csv or .xlsx file."

            if 'Review' not in df_input.columns:
                return "Missing 'Review' column in uploaded file."

            df_input = df_input.dropna(subset=['Review'])
            X = vectorizer.transform(df_input['Review'].astype(str))
            preds = lr_model.predict(X)
            pred_labels = [class_names[p] for p in preds]
            df_input['Sentiment'] = pred_labels

            output_path = os.path.join("static", "predicted_reviews.xlsx")
            df_input.to_excel(output_path, index=False)

            return send_file(output_path, as_attachment=True)

        else:
            return "Please enter a review or upload a file."

    except Exception as e:
        print("Error in prediction:", e)
        return "An error occurred. Please check your input and try again."

@app.route('/hotel-summary', methods=['GET', 'POST'])
def hotel_summary():
    if request.method == 'POST':
        hotel_name = request.form.get('hotel_name')
        if hotel_name not in df['Hotel_Name'].unique():
            return render_template("hotel_summary.html", error="Hotel not found in the dataset.")

        hotel_reviews = df[df['Hotel_Name'] == hotel_name]
        review_text = " ".join(hotel_reviews['Review'].tolist())[:4000]

        tier = hotel_sentiment_summary.loc[hotel_name]['Tier']
        pos_pct = hotel_sentiment_summary.loc[hotel_name]['Positive_Percentage']
        avg_pct = hotel_sentiment_summary[hotel_sentiment_summary['Tier'] == tier]['Positive_Percentage'].mean()
        standing = "above average" if pos_pct > avg_pct else "below average"

        prompt = f"""
You are a hotel benchmarking expert. The hotel name is "{hotel_name}" which belongs to the {tier} tier.
This hotel has a positive review rate of {pos_pct:.2f}%, which is {standing} compared to its tier average of {avg_pct:.2f}%.
Analyze the following guest reviews and return the result in a structured format:

Here are the reviews:
{review_text}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a hospitality analyst generating structured hotel performance evaluations."},
                    {"role": "user", "content": prompt}
                ]
            )
            gpt_summary = response.choices[0].message.content
        except Exception as e:
            return f"GPT API Error: {str(e)}"

        chart_path = generate_radar_chart(hotel_name, hotel_reviews)

        return render_template("hotel_summary.html",
                               hotel_name=hotel_name,
                               tier=tier,
                               pos_pct=round(pos_pct, 2),
                               avg_pct=round(avg_pct, 2),
                               standing=standing,
                               gpt_summary=gpt_summary,
                               chart_path=chart_path)

    return render_template("hotel_summary.html")

if __name__ == "__main__":
    app.run(debug=True)
