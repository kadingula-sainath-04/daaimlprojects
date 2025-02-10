import pandas as pd
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing dataset
dataset = pd.read_csv(r"D:\FULLSTACK DATASCIECE AND AI\Classroomwork\ARTIFICIALINTELLIGENCE\NATURAL LANGUAGE PROCESSING\NLPPTOJECTS\2nd - NLP project\2nd - NLP project\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# Duplicate and combine datasets
duplicate_dataset = dataset.copy()
combined_dataset = pd.concat([dataset, duplicate_dataset], axis=0)
duplicate_dataset_2 = dataset.copy()
final_dataset = pd.concat([combined_dataset, duplicate_dataset_2], axis=0)

corpus = []

# Negation handling preprocessing
def handle_negation(text):
    # Simple negation handling (you can improve this)
    negations = ["not", "never", "n't", "no"]
    words = text.split()
    result = []
    negate = False
    
    for word in words:
        if word in negations:
            negate = True  # Start negating
        elif negate:
            result.append("_NEG_" + word)  # Mark the word after negation
            negate = False  # Stop negating after the next word
        else:
            result.append(word)
    
    return " ".join(result)

# Apply preprocessing for all records
for i in range(0, len(final_dataset)):  
    review = str(final_dataset.iloc[i]['Review'])
    review = re.sub('[^a-zA-Z]', ' ', review)  # Clean the review
    review = review.lower()  # Convert to lowercase
    review = handle_negation(review)  # Apply negation handling
    review = review.split()  # Split into words
    ps = PorterStemmer()  # Initialize the Porter Stemmer
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]  # Stemming and stopword removal
    review = ' '.join(review)  # Join words back into a sentence
    corpus.append(review)

final_dataset['Review'] = corpus

# TF-IDF Vectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()
y = final_dataset.iloc[:, 1].values

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Train LightGBM model
lgbm_model = LGBMClassifier(num_leaves=31, n_estimators=200, min_data_in_leaf=10, max_depth=7, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)

# Confusion matrix and accuracy
cm_lgbm = confusion_matrix(y_test, lgbm_preds)
print(cm_lgbm)

acc_lgbm = accuracy_score(y_test, lgbm_preds)
print(f"Test Accuracy: {acc_lgbm}")

bias_lgbm = lgbm_model.score(X_train, y_train)
print(f"Bias (Training Accuracy): {bias_lgbm}")

variance_lgbm = lgbm_model.score(X_test, y_test)
print(f"Variance (Test Accuracy): {variance_lgbm}")

# Save model and vectorizer
with open("lgbm_model1.pkl", "wb") as model_file:
    pickle.dump(lgbm_model, model_file)

with open("tfidf_vectorizer1.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)
