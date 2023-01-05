### Reviews will have the following categorization: 1 and 2 - Negative, 3- Neutral, 4, 5 - Positive
import matplotlib
# Import the needed packages
import numpy as np
import random
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score  # f1 score to use it as and evaluation metric
import ast  # to convert string into dictionary
from IPython.display import clear_output
from sklearn import svm  # support vector machine classifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression  # import logistic regression
from sklearn.tree import DecisionTreeClassifier  # import Decision tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sb


# Class to categorize a review as positive, negative or neutral

class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE


reviews = []
with open("reviews.txt", encoding="utf-8") as fp:
    for index, line in enumerate(fp):
        review = ast.literal_eval(line)  # this converts the string into a dictionary
        reviews.append(Review(review['reviewBody'], review[
            'stars']))  # this categorizes the review using the Review class and appends it to the reviews.
# print(reviews[0].text)
# print(reviews[0].sentiment)

# Splitting the data: 70% for training and 30% for testing

training, test = train_test_split(reviews, test_size=0.30, random_state=42)

# define the independent variable (x) and target variable (y)

train_x, train_y = [x.text for x in training], [x.sentiment for x in training]
test_x, test_y = [x.text for x in test], [x.sentiment for x in test]

# print("Size of training set data is", len(training))
# print("Size of train set is", len(test))
# print(train_y.count(Sentiment.POSITIVE))
# print(train_y.count(Sentiment.NEGATIVE))
# print(train_y.count(Sentiment.NEUTRAL))
# print(test_y.count(Sentiment.POSITIVE))
# print(test_y.count(Sentiment.NEGATIVE))
# print(test_y.count(Sentiment.NEUTRAL))


def create_unbalanced_data_plot():
    class_ = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    data = [train_y.count("POSITIVE"), train_y.count("NEUTRAL"), train_y.count("NEGATIVE")]
    plt.figure(figsize=(7, 7))
    plt.bar(class_, data)
    plt.title("Imbalanced Data Composition")
    plt.savefig("plots/data_imbalanced.png")
'''
What is bag of words: computer only understands numbers and therefore we
need to convert the review messages we have into a list of numbers using bag-of-words model
Bag-Of-Words: representation of text that describes the occurence of words within a document
Involves: VOCABULARY of knows words and MEASURE OF THE PRESENCE of known words
FREQUENCY OF OCCURRENCE of words is important
'''

# Tokenization

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)
# print(train_x[10])
# print(train_x_vectors[10])
# print(train_x_vectors[10].toarray())


# SVM

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)  # random prediction using SVM

# i = np.random.randint(0, len(test_x))
# print("Review Message: ", test_x[i])
# print("Actual : ", test_y[i])
# print("Prediction: ", clf_svm.predict(test_x_vectors[i]))

# labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
# pred_svm = clf_svm.predict(test_x_vectors)
# cm = confusion_matrix(test_y, pred_svm)
#
# df_cm = pd.DataFrame(cm, index=labels, columns=labels)
#
# sb.heatmap(df_cm, annot=True, fmt='d')  # truth is on "y" and predicted is on "x".
# plt.title("Confusion matrix from SVM [Imbalanced]")
# plt.savefig("./plots/confusion.png")


# Decision Tree
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

# i = np.random.randint(0, 1000)
# print(i)
# print(test_x[i])
# print("Actual:", test_y[i])
# print("Prediction", clf_dec.predict(test_x_vectors[i]))
#
# labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
# pred_dec = clf_dec.predict(test_x_vectors)
# cm = confusion_matrix(test_y, pred_dec)
# df_cm = pd.DataFrame(cm, index=labels, columns=labels)
# sb.heatmap(df_cm, annot=True, fmt='d')
# plt.title("Confusion matrix from Decision Tree [Imbalanced]")
# plt.savefig("plots/confusion_decision_tree.png")


# Logistic Regression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

print(clf_log.predict(test_x_vectors[0]))

# labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
# pred_log = clf_log.predict(test_x_vectors)
# cm = confusion_matrix(test_y, pred_log)
# df_cm = pd.DataFrame(cm, index=labels, columns=labels)
# sb.heatmap(df_cm, annot=True, fmt='d')
# plt.title("Confusion matrix from Logistic Regression [Imbalanced]")
# plt.savefig("plots/confusion_logistic_regression.png")

# Ensemble Random Forest

clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
clf_rf.fit(train_x_vectors, train_y)

# i = np.random.randint(0, 1000)
# print(i)
# print(test_x[i])
# print("Actual:", test_y[i])
# print("Prediction", clf_rf.predict(test_x_vectors[i]))

# labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
# pred_rf = clf_rf.predict(test_x_vectors)
# cm = confusion_matrix(test_y, pred_rf)
# df_cm = pd.DataFrame(cm, index=labels, columns=labels)
# sb.heatmap(df_cm, annot=True, fmt='d')
# plt.title("Confusion matrix from Random Forest [Imbalanced]")
# plt.savefig("plots/confusion_random_forest.png")

classifiers = ["SVM", "DecTree", "LOGISTIC", "RandFor"]

svm_score = clf_svm.score(test_x_vectors, test_y)
decision_tree_score = clf_dec.score(test_x_vectors, test_y)
logistic_score = clf_log.score(test_x_vectors, test_y)
random_forest_score = clf_rf.score(test_x_vectors, test_y)

# scores = [svm_score, decision_tree_score, logistic_score, random_forest_score]
# plt.bar(classifiers, scores)
# plt.title("Accuracy Scores of the Used Classifiers [Imbalanced Data]")
# plt.xlabel("Classifiers")
# plt.ylabel("Accuracy")
# plt.savefig("./plots/accuracy[imbalanced_data].png")
# plt.figure(figsize=(7, 7))
# plt.show()

print(svm_score)
print(decision_tree_score)
print(logistic_score)
print(random_forest_score)
