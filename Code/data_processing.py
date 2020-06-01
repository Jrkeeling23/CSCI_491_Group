import pandas as pd
import os
import nltk
from nltk import WordNetLemmatizer, SnowballStemmer, PorterStemmer
from nltk.corpus import stopwords
from nltk import bigrams
import preprocessor as tweet_preprocessor
import numpy as np
import logging

# check whether the log file is empty or not; if not, clean file
if os.path.exists("data_log.log") and os.stat("data_log.log").st_size != 0:
    old_log = open("data_log.log", "r+")
    content = old_log.read().split("\n")
    old_log.seek(0)
    old_log.truncate()

logging.basicConfig(filename='data_log.log', level=logging.DEBUG)  # logging


class Data:
    def __init__(self):
        self.train_tweet = None  # Train Tweets
        self.train_id = None  # Train ID's
        self.train_label = None  # Train Labels
        self.test_label = None  # Test Labels
        self.test_tweet = None  # Test Tweets
        self.get_files()  # initialize labels ids and tweets to object variables
        # self.train_label = self.train_label[0:10000]
        # self.train_tweet = self.train_tweet[0:10000]
        self.train_label = self.train_label[0:20]
        self.train_tweet = self.train_tweet[0:20]
        self.train_tweet = np.array(self.train_tweet).flatten()  # Sets self.text to a DataFrame

    def get_files(self):
        """
        Get the data files and convert them to pandas DataFrames.
        :return:
        """
        self.train_tweet = list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.text'))  # Get train tweets
        self.train_id = np.array(
            list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.ids')))  # Get train id's
        self.train_label = np.array(
            list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.labels')))  # Get train Labels
        self.test_tweet = list(open('../data/us_test.text'))  # Get test Tweets
        self.test_label = np.array(list(open('../data/us_test.labels')))  # Get test labels

    def stem(self, raw):
        """
        Using PorterStemmer, the tweets are stemmed
        :type raw: list(strings)
        :return: stemmed data, list(list)
        """
        # TODO: update method to return desired data to be stemmed
        logging.info('Stem Data: Begin')
        count = 0
        stemmer = PorterStemmer()  # Initialize a stemmer
        iterator = 0  # Iterator to replace stemmed sentence back into self.text list
        for row in self.train_tweet:  # Loops through each tweet in self.text
            stemmed_list = []  # Temp list to add back into self.text
            split_row = row.split()  # Splits the tweet into words by ' '.
            count += 1
            for word in split_row:  # Loops through each word
                stemmed_list.append(stemmer.stem(word))  # Appends the stemmed word to the temp list.
            self.train_tweet[iterator] = ' '.join(
                stemmed_list)  # converts the list to a sentence ands adds back into self.text
            iterator += 1
        logging.info('Stem Data: Complete')

    def bigrams(self, raw):
        """
        Create bigrams from the specified data
        :param raw: tweets train or test
        :return: bigrams created
        """
        logging.info('Bigrams: Begin')
        bigram = []
        for tweet in raw:
            bigram.append(list(nltk.bigrams(tweet.split())))
        logging.info('Bigrams: Complete')
        return bigram


    def lemmatize(self):  # Method to lemmatize test
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
        # Source for tweet cleaning: https://pypi.org/project/tweet-preprocessor/
        iterator = 0
        for row in self.train_tweet:  # Loops through each tweet
            self.train_tweet[iterator] = tweet_preprocessor.clean(
                row)  # Replaces self.text tweet with the cleaned tweed.
            iterator += 1

    def stop_words(self):  # Removes stop words from tweet.
        # Following stop words code sourced from : https://pythonspot.com/nltk-stop-words/
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
