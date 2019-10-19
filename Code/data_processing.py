import pandas as pd
import nltk
from nltk import WordNetLemmatizer, SnowballStemmer, LancasterStemmer
import preprocessor as tweet_preprocessor


class Data():
    def __init__(self):
        self.text = None  # Text DataFrame
        self.id = None  # ID DataFrame
        self.label = None  # Label DataFrame
        self.get_files()
        self.text = self.text[0:20] #TODO use full data set
        self.remove_punctuation()
        # self.stem()
        self.lemmatize()
        for i in self.text:
            print(i)

    def get_files(self):  # Get the data files and convert them to pandas DataFrames.
        print('\n----------------------------------------------Getting Data Files----------------------------------------------\n')
        self.text = list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.text'))  # Makes Text a list
        self.id = pd.DataFrame(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.ids'))
        self.label = pd.DataFrame(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.labels'))

    def stem(self):  # Method to stem test
        print('\n----------------------------------------------Stemming Data----------------------------------------------\n')
        stemmer = LancasterStemmer()
        iterator = 0
        for row in self.text:
            stemmed_list = []
            split_row = row.split()
            for word in split_row:
                stemmed_list.append(stemmer.stem(word))
            self.text[iterator] = ' '.join(stemmed_list)
            iterator += 1

    def lemmatize(self):  # Method to stem test
        nltk.download('wordnet')
        print('\n----------------------------------------------Stemming Data----------------------------------------------\n')
        lemmatizer = WordNetLemmatizer()
        iterator = 0
        for row in self.text:
            lemmatized_list = []
            split_row = row.split()
            for word in split_row:
                lemmatized_list.append(lemmatizer.lemmatize(word))
            self.text[iterator] = ' '.join(lemmatized_list)
            iterator += 1

    def remove_punctuation(self):
        print('\n----------------------------------------------Cleaning Tweets----------------------------------------------\n')
        # Source for tweet cleaning: https://pypi.org/project/tweet-preprocessor/
        iterator = 0
        for row in self.text:
            self.text[iterator] = tweet_preprocessor.tokenize(row)
            iterator += 1
