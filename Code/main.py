from data_processing import Data
from algorithm import Algorithm
import pandas as pd


class Main():

    def __init__(self):
        self.data = Data()  # Contains id, text, label variables
        self.algorithm = Algorithm(self.data) # Contains Naive Bayes, KNN

        # Run machine learning algorithms
        self.algorithm.naive_bays()
        heighest_acc = 0.0
        tuned_neighbors = 0
        for k_neighbors in range(1, 25):
            acc = self.algorithm.KNN(k_neighbors)
            print("Accuracy ", acc)
            if acc > heighest_acc:
                heighest_acc = acc
                tuned_neighbors = k_neighbors
        print("Highest accuracy neighbor count: ", tuned_neighbors, " with accuracy ", heighest_acc)
if __name__ == "__main__":
    Main()
