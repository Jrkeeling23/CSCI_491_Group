from data_processing import Data
from algorithm import Algorithm
import pandas as pd


class Main():

    def __init__(self):
        self.data = Data()  # Contains id, text, label variables
        self.algorithm = Algorithm(self.data) # Contains Naive Bayes, KNN

        # Run machine learning algorithms
        self.algorithm.naive_bays()
        for k_neighbors in range(5, 15, 5):
            self.algorithm.KNN(k_neighbors)


if __name__ == "__main__":
    Main()
