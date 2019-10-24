import pickle

from sklearn import metrics, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from os import path
from sklearn.linear_model import SGDClassifier


class Algorithm:
    def __init__(self, data, tune_params):
        self.data = data  # @Variables: Id, Text, Label
        # TODO test on other data

        self.vectorizer = CountVectorizer('english', lowercase=True)
        self.data_vector = self.vectorizer.fit_transform(self.data.train_tweet)  # Fit a fector on the train data
        self.data_vector_test = self.vectorizer.transform(self.data.test_tweet)  # Fit the test data into a vector
        if tune_params:
            # self.tune_naive_bayes()
            # self.tune_KNN()
            # self.tune_svm_svc()
            self.tune_svm_linear()

    def naive_bayes(self):  # Predict with Naive Bayes
        self.data.print_title('Naive Bayes')
        if not self.model_exists('NB.sav'):  # Saves the model if it doesn't exist
            # Gets the data vector for the algorithm. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
            # tuned_params = self.tune_naive_bayes()  # Tune Parameters
            # nb_model = MultinomialNB(alpha=tuned_params['alpha'],
            #                          fit_prior=tuned_params['fit_prior'])  # Instantiate the Naive Bayes Model
            nb_model = MultinomialNB(alpha=0.7,
                                     fit_prior=True)  # Instantiate the Naive Bayes Model
            self.data.print_title('Fitting Naive Bayes Model')
            nb_model.fit(self.data_vector, self.data.train_label)  # Fit the naive bayes model
            self.save_model(nb_model, 'NB.sav')

        nb_model = self.load_model('NB.sav')  # Load the model
        self.data.print_title('Predicting Naive Bayes')

        predict = nb_model.predict(self.data_vector_test)  # Predict with naive bayes

        # Following print statement. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
        print("\n", metrics.classification_report(self.data.test_label, predict))  # Print Metrics

    def KNN(self):  # Method for K nearest neightbors
        self.data.print_title(' K Nearest Neighbors')
        if not self.model_exists('KNN.sav'):  # Saves the model if it doesn't exist
            # tuned_params = self.tune_KNN()  # Tune Parameters
            # knn_model = KNeighborsClassifier(n_neighbors=tuned_params['n_neighbors'], weights=tuned_params['weights'],
            #                                  algorithm=tuned_params['algorithm'], p=tuned_params['p'],
            #                                  metric=tuned_params['metric'])  # Instantiate K Nearest Neighbors model
            knn_model = KNeighborsClassifier(n_neighbors=26, weights='uniform', algorithm='auto', p='p',
                                             metric='cosine')  # Instantiate K Nearest Neighbors model
            self.data.print_title('Fitting KNN Model')
            knn_model.fit(self.data_vector, self.data.train_label)  # Fit the data to the model
            self.save_model(knn_model, 'KNN.sav')
        knn_model = self.load_model('KNN.sav')  # Load the model
        self.data.print_title('Predicting KNN')
        predict = knn_model.predict(self.data_vector_test)  # Predict on the test set
        # Following print statement. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
        # print("\n", metrics.classification_report(self.y_test, predict))  # Print Metrics
        print("\n", metrics.classification_report(self.data.test_label, predict))  # Print Metrics

    def SVM_linear(self):  # Method for support vector machine Linear
        # Following svm code sourced from: https://scikit-learn.org/stable/modules/svm.html
        self.data.print_title('SVM Linear')
        if not self.model_exists('SVM_linear.sav'):  # Saves the model if it doesn't exist
            # tuned_params = self.tune_svm_linear()  # Tune Parameters
            # svm_model = svm.LinearSVC(C=tuned_params['C'], dual=tuned_params['dual'],
            #                           fit_intercept=tuned_params['fit_intercept'],
            #                           multi_class=tuned_params['multi_class'],
            #                           penalty=tuned_params['penalty'])  # Instantiate SVM Model

            svm_model = SGDClassifier()  # Instantiate SVM Model

            # svm_model = svm.LinearSVC(C=0.1, dual=False, fit_intercept=True, multi_class='ovr', penalty='l1')  # Instantiate SVM Model
            self.data.print_title('Fitting SVM Linear Model')
            svm_model.fit(self.data_vector, self.data.train_label)  # Fit the data to the model
            self.save_model(svm_model, 'SVM_linear.sav')
        svm_model = self.load_model('SVM_linear.sav')  # Load the model
        self.data.print_title('Predicting SVM')
        predict = svm_model.predict(self.data_vector_test)  # Predict on the test set
        # Following print statement. Source from assignment 3: https://colab.research.google.com/drive/1QjU4Y306pfmAozerZwrLvtaBUhJOCZFz#scrollTo=_ru8k_nK05xu
        # print("\n", metrics.classification_report(self.y_test, predict))  # Print Metrics
        print("\n", metrics.classification_report(self.data.test_label, predict))  # Print Metrics

    def model_exists(self, model):  # Checks if the model exists
        # Following code sourced from : https://www.guru99.com/python-check-if-file-exists.html
        model_path = '../models/' + model
        return path.exists(model_path)  # Returns boolean

    def save_model(self, model, file_name):  # Saves the model for the algorithm
        self.data.print_title('Saving Model')

        # Source to save model: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
        file_name = '../models/' + file_name
        pickle.dump(model, open(file_name, 'wb'))

    def load_model(self, model_file):  # Loads the model for the algorithm
        self.data.print_title('Loading Model')
        # Source to load model: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
        model_path = '../models/' + model_file
        return pickle.load(open(model_path, 'rb'))

    def tune_naive_bayes(self):  # Tunes Naive Bayes
        self.data.print_title('Tuning Naive Bayes')

        # Following code for tuning sourced from: https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
        params = {'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'fit_prior': [True, False]}  # Set the params and values to tune
        grid_search = GridSearchCV(MultinomialNB(), params, verbose=3)  # Instantiate grid search
        grid_search.fit(self.data_vector, self.data.train_label)  # Fit the best parameters
        print("NB best params: ", grid_search.best_params_)
        return grid_search.best_params_

    def tune_KNN(self):  # Tune the KNN parameters
        self.data.print_title('Tuning KNN')

        # Following code for tuning sourced from: https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

        params = {'n_neighbors': list(range(5, 30)), 'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'brute'], 'p': ['p', 1, 2],
                  'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                             'manhattan']}  # Set the params and values to tune
        grid_search = GridSearchCV(KNeighborsClassifier(), params, verbose=3)  # Instantiate grid search
        grid_search.fit(self.data_vector, self.data.train_label)  # Fit the best parameters
        print("KNN best params: ", grid_search.best_params_)
        return grid_search.best_params_

    def tune_svm_linear(self):  # Tune svm_Linear params
        self.data.print_title('Tuning Linear SVM')

        # Following code for tuning sourced from: https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

        params = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'penalty': ['l1', 'l2'],
                  'dual': [False],
                  'multi_class': ['ovr', 'crammer_singer'],
                  'fit_intercept': [True, False]}  # Set the params and values to tune
        grid_search = GridSearchCV(svm.LinearSVC(), params, verbose=3)  # Instantiate grid search
        grid_search.fit(self.data_vector, self.data.train_label)  # Fit the best parameters
        print("Linear SVM best params: ", grid_search.best_params_)
        return grid_search.best_params_
