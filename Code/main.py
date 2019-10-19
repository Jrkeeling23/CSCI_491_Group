from data_processing import Data
from algorithm import Algorithm
import pandas as pd


class Main():

    def __init__(self):
        self.data = Data()  # Contains id, text, label variables
        self.algorithm = Algorithm(self.data)

if __name__ == "__main__":
    Main()
