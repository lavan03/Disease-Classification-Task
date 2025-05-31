
# 🧠 Medical Text Classification

This project builds a text classification model to predict medical disease categories from clinical descriptions. It uses natural language preprocessing (SpaCy), TF-IDF feature extraction, and machine learning classifiers like Logistic Regression, LinearSVC, and Multinomial Naive Bayes.

---

## 📁 Dataset

The dataset contains 316 samples with the following columns:

- `disease`: Name of the disease (e.g., "Asthma")
- `description`: Raw clinical description text
- `category`: Target label indicating the medical system (e.g., "respiratory", "neurologic")

---

## 🔧 Preprocessing

Text is cleaned using:
- Lowercasing
- Tokenization and lemmatization (via SpaCy)
- Removal of stopwords, medical domain-specific terms, adjectives, punctuation, and numbers


preprocess_text_spacy(text):
    # SpaCy pipeline that removes stopwords, adjectives, etc.


A new column `description_processed` is created in the DataFrame for model training.

---

## 🔍 Feature Extraction

* **TF-IDF Vectorization** with bi-grams:
TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)




## 🤖 Models Used

We evaluate 3 models:

| Model              | Description                                     |
| ------------------ | ----------------------------------------------- |
| LogisticRegression | Baseline classifier with class weight balancing |
| LinearSVC          | High-performing linear SVM for text             |
| MultinomialNB      | Fast and simple for short texts                 |

All models are evaluated using accuracy and macro-averaged F1 scores.



## 📊 Evaluation Metrics

Example performance (Logistic Regression):


Accuracy: 76%

Classification Report:
- Good performance on majority classes
- Some underperformance on minority categories


## 📈 Future Improvements

* Use **SMOTE** or data augmentation for rare classes
* Try **transformer-based embeddings** (e.g., BioBERT)
* Add hyperparameter tuning with `GridSearchCV`
* Include more domain-specific cleaning and entity recognition


