import pandas as pd
import nltk
from nltk import WordNetLemmatizer, SnowballStemmer, LancasterStemmer
from nltk.corpus import stopwords
import preprocessor as tweet_preprocessor
import numpy as np


class Data():
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.train_tweet = None  # Text DataFrame
        self.train_id = None  # ID DataFrame
        self.train_label = None  # Label DataFrame
        self.get_files()
        # self.text = self.text[0:100000]  # TODO use full data set
        # self.id = self.id[0:100000]   # TODO use full data set
        # self.label = self.label[0:100000]   # TODO use full data set
        # self.stop_words()
        # self.remove_punctuation()
        # self.stem()
        # self.lemmatize()
        # Numpy flatten sourced from https://stackoverflow.com/questions/47675520/getting-error-on-standardscalar-fit-transform , user:O.Suleiman
        self.train_tweet = np.array(self.train_tweet).flatten()  # Sets self.text to a DataFrame

    def get_files(self):  # Get the data files and convert them to pandas DataFrames.
        self.print_title('Getting Data Files')

        self.train_tweet = list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.text'))  # Makes Text a list
        # Numpy flatten sourced from https://stackoverflow.com/questions/47675520/getting-error-on-standardscalar-fit-transform , user:O.Suleiman
        self.train_id = np.array(list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.ids'))).flatten()
        self.train_label = np.array(list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.labels'))).flatten()
        self.test_label = np.array(list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.labels'))).flatten()
    def stem(self):  # Method to stem test
        self.print_title('Stemming Data')

        stemmer = LancasterStemmer()  # Initialize a stemmer
        iterator = 0  # Iterator to replace stemmed sentence back into self.text list
        for row in self.train_tweet:  # Loops through each tweet in self.text
            stemmed_list = []  # Temp list to add back into self.text
            split_row = row.split()  # Splits the tweet into words by ' '.
            for word in split_row:  # Loops through each word
                stemmed_list.append(stemmer.stem(word))  # Appends the stemmed word to the temp list.
            self.train_tweet[iterator] = ' '.join(
                stemmed_list)  # converts the list to a sentence ands adds back into self.text
            iterator += 1

    def lemmatize(self):  # Method to lemmatize test
        self.print_title('Lemmatizing Data')

        lemmatizer = WordNetLemmatizer()  # Instantiate a lemmatizer
        iterator = 0
        for row in self.train_tweet:  # Loos through each iteration tweet in self.text
            lemmatized_list = []  # Temp list to add back into self.text after tweet is lemmatized
            split_row = row.split()  # Splits the tweet into individual words
            for word in split_row:  # loops through each word.
                lemmatized_list.append(lemmatizer.lemmatize(word))  # Appends the lemmatized word into a temp list.
            self.train_tweet[iterator] = ' '.join(lemmatized_list)  # Replaces tweet with a lemmatized tweet.
            iterator += 1

    def remove_punctuation(self):  # Removes tweet punctuation.
        self.print_title('Cleaning Tweets')

        # Source for tweet cleaning: https://pypi.org/project/tweet-preprocessor/
        iterator = 0
        for row in self.train_tweet:  # Loops through each tweet
            self.train_tweet[iterator] = tweet_preprocessor.clean(row)  # Replaces self.text tweet with the cleaned tweed.
            iterator += 1

    def stop_words(self):  # Removes stop words from tweet.
        # Following stop words code sourced from : https://pythonspot.com/nltk-stop-words/

        self.print_title('Removing Stop Words From Data')
        iterator = 0
        for row in self.train_tweet:  # Loops through each tweet in self.text
            stop_word_list = []  # Temp list to append tweet without stop words into self.text
            split_row = row.split()  # Splits the tweet into individual words.
            for word in split_row:  # Loops through each word
                if word not in set(stopwords.words('english')):  # If word not in nltk's stop words list.
                    stop_word_list.append(word)  # Append the word to a temp list.
            self.train_tweet[iterator] = ' '.join(
                stop_word_list)  # Turn the list back into a string and replace self.text location.
            iterator += 1

    def print_title(self, string): # Prints the string to the approx same length
        dash_length = 100 - len(string)
        dashes = '-' * (int(dash_length / 2))
        new_string = (dashes + ' ' + string + ' '+ dashes)
        print(new_string)
