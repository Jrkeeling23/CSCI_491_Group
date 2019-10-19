from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class Algorithm:
    def __init__(self, data):
        self.data = data  # @Variables: Id, Text, Label
        # TODO test on other data
        # Train test split sourced from: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data.text, self.data.label,
                                                                                test_size=0.2)  # Splits the data into train and test sets
        self.vectorizer = CountVectorizer('english')
        self.data_vector = self.vectorizer.fit_transform(self.x_train)  # Fit a fector on the train data
        self.data_vector_test = self.vectorizer.transform(self.x_test)  # Fit the test data into a vector

        self.naive_bays()
        self.KNN(5)
        self.KNN(10)

    def naive_bays(self):  # Predict with Naive Bayes

        print(
            '\n----------------------------------------------Naive Bayes----------------------------------------------\n')

        # Gets the data vector for the algorithm. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
        model = MultinomialNB()  # Instantiate the Naive Bayes Model
        model.fit(self.data_vector, self.y_train)  # Fit the naive bayes model
        predict = model.predict(self.data_vector_test)  # Predict with naive bayes

        # Following print statement. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
        print("\n", metrics.classification_report(self.y_test, predict)) # Print Metrics

    def KNN(self, k_neighbors): # Method for K nearest neightbors
        print(
            '\n----------------------------------------------', k_neighbors, ' Nearest Neighbors----------------------------------------------\n')

        model = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance') # Instantiate K Nearest Neighbors model
        model.fit(self.data_vector, self.y_train) # Fit the data to the model
        predict = model.predict(self.data_vector_test) # Predict on the test set
       # Following print statement. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
        print("\n", metrics.classification_report(self.y_test, predict)) # Print Metrics
