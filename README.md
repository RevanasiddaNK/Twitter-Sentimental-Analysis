# Twitter Sentiment Analysis 🐦📊

This project performs sentiment analysis on tweets using classical machine learning models. It uses **TF-IDF** for feature extraction, trains **Naive Bayes** and **Logistic Regression** models, and combines them with a **VotingClassifier** (soft voting). The model is then used to predict sentiment and visualize trends over time.

---

## 📁 Project Structure

```

├── train_model.ipynb           # Training notebook for TF-IDF + VotingClassifier
├── predict_model.ipynb         # Load model and predict sentiment on new text
├── sentiment_trend_plot.ipynb  # Visualization of sentiment trends using matplotlib
├── requirements.txt            # Dependencies
├── saved_model.pkl             # Trained pipeline model (TF-IDF + Voting)
├── twitter_data.csv            # Input dataset (0 = Negative, 4 = Positive)

````

---

## ✅ Features

- **TF-IDF Vectorization**: Converts text data into numerical features.
- **MultinomialNB**: Naive Bayes classifier for text classification.
- **LogisticRegression**: Another classical model for comparison and ensemble.
- **VotingClassifier (soft)**: Combines predictions using averaged probabilities.
- **Pipeline**: End-to-end model (TF-IDF + VotingClassifier) packed and saved.
- **Sentiment Prediction**: Predict sentiment (0 = Negative, 4 = Positive) for new tweets.
- **Trend Visualization**: Monthly sentiment trend plotted using `matplotlib`.

---

## 📊 Dataset

The dataset used is `twitter_data.csv`, which contains tweets labeled with:
- `0`: Negative sentiment  
- `4`: Positive sentiment

Ensure the dataset contains at least:
- `text`: The tweet content
- `targett`: Sentiment label (0 or 4)
- `date`: Date of tweet (used in trend analysis)

---

## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
````

### 2. Train the Model

Open and run `train_model.ipynb` to:

* Vectorize text
* Train `MultinomialNB` and `LogisticRegression`
* Combine with `VotingClassifier`
* Save the model as `saved_model.pkl`

### 3. Predict Sentiment

Open `predict_model.ipynb` to:

* Load `saved_model.pkl`
* Predict sentiment for new inputs

### 4. Visualize Trends

Open `sentiment_trend_plot.ipynb` to:

* Plot sentiment trend over months using `matplotlib`

---

## 📌 Requirements

* numpy
* pandas
* scikit-learn
* matplotlib
* joblib

(Install via `requirements.txt`)

---

## 📷 Sample Output

* 📈 Monthly sentiment trends for tweets (Positive vs Negative)
* ✅ Console output showing accuracy and classification report

---

## 🧠 Model Insight

The combination of Naive Bayes and Logistic Regression allows us to leverage:

* Fast & robust performance of Naive Bayes on sparse text
* Generalization capabilities of Logistic Regression
  Soft voting combines them by averaging prediction probabilities.

---

## ✍️ Author

**Revanasidda N Karigoudar**
AI & ML Engineer | Full-Stack Developer
[GitHub](https://github.com/RevanasiddaNK)

---

## 📜 License

This project is licensed for educational purposes.

```

---

Let me know if you'd like badges (e.g., GitHub stars, Python version), or a markdown preview version.
```
