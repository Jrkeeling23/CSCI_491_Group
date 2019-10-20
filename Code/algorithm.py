import pickle

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from os import path


class Algorithm:
    def __init__(self, data):
        self.data = data  # @Variables: Id, Text, Label
        # TODO test on other data

        self.vectorizer = CountVectorizer('english')
        self.data_vector = self.vectorizer.fit_transform(self.data.train_tweet)  # Fit a fector on the train data
        self.data_vector_test = self.vectorizer.transform(self.data.test_tweet)  # Fit the test data into a vector

    def naive_bays(self):  # Predict with Naive Bayes
        self.data.print_title('Naive Bayes')
        if not self.model_exists('NB.sav'): # Saves the model if it doesn't exist
            # Gets the data vector for the algorithm. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
            nb_model = MultinomialNB()  # Instantiate the Naive Bayes Model
            nb_model.fit(self.data_vector, self.data.train_label)  # Fit the naive bayes model
            self.save_model(nb_model, 'NB.sav')

        nb_model = self.load_model('NB.sav') # Load the model
        self.data.print_title('Predicting Naive Bayes')

        predict = nb_model.predict(self.data_vector_test)  # Predict with naive bayes

        # Following print statement. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
        print("\n", metrics.classification_report(self.data.test_label, predict))  # Print Metrics

    def KNN(self, k_neighbors):  # Method for K nearest neightbors
        self.data.print_title(str(str(k_neighbors) + ' Nearest Neighbors'))
        if not self.model_exists('KNN.sav'): # Saves the model if it doesn't exist
            knn_model = KNeighborsClassifier(n_neighbors=k_neighbors,
                                         weights='distance')  # Instantiate K Nearest Neighbors model
            knn_model.fit(self.data_vector, self.data.train_label)  # Fit the data to the model
            self.save_model(knn_model, 'KNN.sav')
        knn_model = self.load_model('KNN.sav') # Load the model
        self.data.print_title('Predicting KNN')
        predict = knn_model.predict(self.data_vector_test)  # Predict on the test set
        # Following print statement. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
        # print("\n", metrics.classification_report(self.y_test, predict))  # Print Metrics
        print("\n", metrics.classification_report(self.data.test_label, predict))  # Print Metrics

    def model_exists(self, model):  # Checks if the model exists
        # Following code sourced from : https://www.guru99.com/python-check-if-file-exists.html
        model_path = '../models/' + model
        return path.exists(model_path)  # Returns boolean

    def save_model(self, model, file_name): # Saves the model for the algorithm
        self.data.print_title('Saving Model')

        # Source to save model: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
        file_name = '../models/' + file_name
        pickle.dump(model, open(file_name, 'wb'))

    def load_model(self, model_file): # Loads the model for the algorithm
        self.data.print_title('Loading Model')
        # Source to load model: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
        model_path = '../models/' + model_file
        return pickle.load(open(model_path, 'rb'))
