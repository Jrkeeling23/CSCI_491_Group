import pandas as pd


class Data():
    def __init__(self):
        self.text = None  # Text DataFrame
        self.id = None  # ID DataFrame
        self.label = None  # Label DataFrame
        self.get_files()

    def get_files(self):  # Get the data files and convert them to pandas DataFrames.
        self.text = pd.DataFrame(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.text'))
        self.id = pd.DataFrame(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.ids'))
        self.label = pd.DataFrame(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.labels'))
