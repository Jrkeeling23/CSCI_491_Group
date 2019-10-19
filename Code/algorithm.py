from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


class Algorithm:
    def __init__(self, data):
        self.data = data  # @Variables: Id, Text, Label
        # TODO test on other data
        # Train test split sourced from: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data.text, self.data.label,
                                                                                test_size=0.2)  # Splits the data into train and test sets
        self.vectorizer = CountVectorizer('english')
        self.naive_bays()

    def naive_bays(self):  # Predict with Naive Bayes

        print(
            '\n----------------------------------------------Naive Bayes----------------------------------------------\n')

        # Gets the data vector for the algorithm. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
        data_vector = self.vectorizer.fit_transform(self.x_train)  # Fit a fector on the train data
        model = MultinomialNB()  # Instantiate the Naive Bayes Model
        model.fit(data_vector, self.y_train)  # Fit the naive bayes model
        data_vector_test = self.vectorizer.transform(self.x_test)  # Fit the test data into a vector
        predict = model.predict(data_vector_test)  # Predict with naive bayes

        # Following print statement. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
        print("macro F1:", metrics.f1_score(self.y_test, predict, average='macro'))
        print("micro F1:", metrics.f1_score(self.y_test, predict, average='micro'))
        print("\n", metrics.classification_report(self.y_test, predict))
        cm = metrics.confusion_matrix(self.y_test, predict)
        print("Confusion Matrix:\n", cm)
