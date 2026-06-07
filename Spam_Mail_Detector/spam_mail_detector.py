import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    'message': [
        'Win a free iPhone now',
        'Congratulations you won a prize',
        'Meeting at 5 PM today',
        'Please submit your assignment',
        'Claim your free reward now',
        'Project discussion tomorrow'
    ],
    'label': [
        'spam',
        'spam',
        'ham',
        'ham',
        'spam',
        'ham'
    ]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)