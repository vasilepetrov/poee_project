from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "Excellent Services by the ABC remit team.Recommend.",
    "Bad Services. Transaction delayed for three days.Don't recommend.",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out()) # this gives the vocabulary
print(X.toarray())