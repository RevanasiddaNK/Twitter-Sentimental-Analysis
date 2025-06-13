
---

```markdown
# 🐦 Twitter Sentiment Analysis

A machine learning project that performs sentiment analysis on 1.6M+ tweets using the [Sentiment140](http://help.sentiment140.com/for-students/) dataset. It classifies tweets as **positive** or **negative** using both Logistic Regression and Naive Bayes models, with visualization of sentiment trends over time.

---

## 📊 Demo Results

| Model              | Accuracy (%) |
|--------------------|--------------|
| Logistic Regression (TF-IDF) | 79.3% |
| Naive Bayes (TF-IDF + Bigrams) | 78.1% |

---

## 🧠 Features

- 🔄 Preprocessing: Cleaned tweets by removing usernames, links, punctuation, and stopwords.
- 🧮 Feature Extraction: TF-IDF and Bigrams using `TfidfVectorizer`.
- 🏁 ML Models: Trained Logistic Regression and Multinomial Naive Bayes.
- 📈 Visualized sentiment trends over time using `matplotlib`.
- ✅ Model Evaluation with accuracy scores and sample predictions.
- 💾 Trained models stored as `.sav` files (external download link).

---

## 📂 Project Structure

```

Twitter-Sentimental-Analysis/
├── data/
│   └── sentiment140.csv        # Raw or cleaned tweet data
├── train\_model.ipynb           # Jupyter notebook for training and evaluation
├── predict\_model.ipynb         # Notebook to test predictions from saved models
├── requirements.txt            # Required libraries
├── README.md                   # This file
└── .gitignore

````

---

## 🔍 Dataset

- **Source:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size:** 1.6 million tweets
- **Format:**
  - `0`: Negative
  - `4`: Positive
  - Other columns: Tweet ID, date, query, username, tweet text

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/RevanasiddaNK/Twitter-Sentimental-Analysis.git
cd Twitter-Sentimental-Analysis
````

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Model Training

Run the training notebook:

```bash
jupyter notebook train_model.ipynb
```

This:

* Cleans the tweets
* Extracts features using `TfidfVectorizer` with or without bigrams
* Trains:

  * Logistic Regression (TF-IDF)
  * Naive Bayes (TF-IDF + Bigrams)
* Saves models as `.sav` files

---

## 🧪 Model Testing

Run the prediction notebook:

```bash
jupyter notebook predict_model.ipynb
```

It loads the saved models and:

* Predicts sentiment for individual tweets
* Prints actual vs predicted values

---

## 📈 Sentiment Trend Visualization

```python
import matplotlib.pyplot as plt

# Visualize number of positive and negative tweets by day
df['date'] = pd.to_datetime(df['date'])
df['sentiment'] = df['target'].replace({0: "Negative", 4: "Positive"})
df['day'] = df['date'].dt.date

plt.figure(figsize=(12,6))
df.groupby(['day', 'sentiment']).size().unstack().plot(kind='line', linewidth=2, marker='o')
plt.title("📊 Sentiment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Tweet Count")
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## 💾 Model Downloads

Due to GitHub's 100 MB limit, trained models are hosted externally:

* [🔗 Logistic Regression Model (`lr_trained_model.sav`)](https://drive.google.com/your-link-here)
* [🔗 Naive Bayes Model (`nb_trained_model.sav`)](https://drive.google.com/your-link-here)

You can use `gdown` to download in code:

```python
!pip install gdown
!gdown --id your_file_id_here
```

---

## ✅ Evaluation

Example output:

```
📌 Model: Logistic Regression
🔎 Predicting sentiment for tweet #400
✅ Actual Sentiment   : Negative
🤖 Predicted Sentiment: Positive
```

---

## 📦 Dependencies

```
numpy
pandas
scikit-learn
matplotlib
nltk
tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

* Add real-time Twitter API integration
* Build a web app using Flask/Streamlit
* Improve accuracy with deep learning models like LSTM/BERT
* Export confusion matrix, ROC curves

---

## 🙋‍♂️ Author

**Revanasidda N Karigoudar**
📧 [43.revanasidda@gmail.com](mailto:43.revanasidda@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/revanasiddan/)
🔗 [GitHub](https://github.com/RevanasiddaNK)

---

## 📜 License

MIT License – free to use and modify.

```

---

Let me know if you'd like me to include a download script for models or set this up as a `README.md` file for direct copy-paste.
```
