from data_processing import Data
from algorithm import Algorithm
import pandas as pd
import matplotlib.pyplot as plt


class Main():

    def __init__(self):
        self.data = Data()  # Contains id, text, label variables
        tune_algo_params = True
        self.algorithm = Algorithm(self.data, tune_algo_params)  # Contains Naive Bayes, KNN
        # Run machine learning algorithms
        # self.algorithm.naive_bayes()
        # self.algorithm.SVM_linear()
        # self.algorithm.KNN()

        # self.algorithm.SVM()


if __name__ == "__main__":
    Main()
