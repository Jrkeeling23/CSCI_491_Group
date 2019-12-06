import pandas as pd
import nltk
from nltk import WordNetLemmatizer, SnowballStemmer, LancasterStemmer
from nltk.corpus import stopwords
import preprocessor as tweet_preprocessor
import numpy as np
from textblob import TextBlob


class Data:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.train_tweet = None  # Train Tweets
        self.train_id = None  # Train ID's
        self.train_label = None  # Train Labels
        self.test_label = None # Test Labels
        self.test_tweet = None # Test Tweets
        self.get_files()
        # self.train_label = self.train_label[0:10000]
        # self.train_tweet = self.train_tweet[0:10000]

        # self.train_tweet = self.get_sentiment(self.train_tweet)
        # self.test_tweet = self.get_sentiment(self.test_tweet)

        # self.train_tweet = self.stop_words(self.train_tweet)
        # self.test_tweet = self.stop_words(self.test_tweet)
        # self.train_tweet = self.tweet_processor(self.train_tweet)
        # self.test_tweet = self.tweet_processor(self.test_tweet)
        # self.train_tweet = self.remove_punc(self.train_tweet, self.train_sentiment)
        # self.test_tweet = self.remove_punc(self.test_tweet, self.test_sentiment)
        # self.train_tweet = self.stem(self.train_tweet)
        # self.test_tweet = self.stem(self.test_tweet)
        # self.train_tweet = self.lemmatize(self.train_tweet)
        # self.test_tweet = self.lemmatize(self.test_tweet)
        # self.remove_punctuation()
        # self.stem()
        # self.lemmatize()
        # Numpy flatten sourced from https://stackoverflow.com/questions/47675520/getting-error-on-standardscalar-fit-transform , user:O.Suleiman
        self.train_tweet = np.array(self.train_tweet)  # Sets self.text to a DataFrame
        self.test_tweet = np.array(self.test_tweet)
# TODO Deal with potential missing data


    def get_files(self):  # Get the data files and convert them to pandas DataFrames.
        self.print_title('Getting Data Files')
        # self.slang = list(open('../SlangSD/SlangSD.txt'))
        self.train_tweet = list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.text'))  # Get train tweets
        # Numpy flatten sourced from https://stackoverflow.com/questions/47675520/getting-error-on-standardscalar-fit-transform , user:O.Suleiman
        self.train_id = np.array(
            list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.ids'))).flatten()  # Get train id's
        self.train_label = np.array(
            list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.labels'))).flatten()  # Get train Labels
        self.test_tweet = list(open('../data/us_test.text'))  # Get test Tweets
        self.test_label = list(open('../data/us_test.labels'))  # Get test labels

    def stem(self, tweet):  # Method to stem test
        self.print_title('Stemming Data')

        stemmer = LancasterStemmer()  # Initialize a stemmer
        iterator = 0  # Iterator to replace stemmed sentence back into self.text list
        for row in tweet:  # Loops through each tweet in self.text
            stemmed_list = []  # Temp list to add back into self.text
            split_row = row.split()  # Splits the tweet into words by ' '.
            for word in split_row:  # Loops through each word
                stemmed_list.append(stemmer.stem(word))  # Appends the stemmed word to the temp list.
            tweet[iterator] = ' '.join(
                stemmed_list)  # converts the list to a sentence ands adds back into self.text
            iterator += 1
        return tweet

    def lemmatize(self, tweet):  # Method to lemmatize test
        self.print_title('Lemmatizing Data')

        lemmatizer = WordNetLemmatizer()  # Instantiate a lemmatizer
        iterator = 0
        for row in tweet:  # Loos through each iteration tweet in self.text
            lemmatized_list = []  # Temp list to add back into self.text after tweet is lemmatized
            split_row = row.split()  # Splits the tweet into individual words
            for word in split_row:  # loops through each word.
                lemmatized_list.append(lemmatizer.lemmatize(word))  # Appends the lemmatized word into a temp list.
            tweet[iterator] = ' '.join(lemmatized_list)  # Replaces tweet with a lemmatized tweet.
            iterator += 1
        return tweet

    def tweet_processor(self, tweet):  # Removes tweet punctuation.
        self.print_title('Cleaning Tweets')

        # Source for tweet cleaning: https://pypi.org/project/tweet-preprocessor/
        iterator = 0
        for row in tweet:  # Loops through each tweet
            tweet[iterator] = tweet_preprocessor.tokenize(
                row)  # Replaces self.text tweet with the cleaned tweed.
            iterator += 1
        return tweet

    def stop_words(self, tweet):  # Removes stop words from tweet.
        # Following stop words code sourced from : https://pythonspot.com/nltk-stop-words/

        self.print_title('Removing Stop Words From Data')
        iterator = 0
        for row in tweet:  # Loops through each tweet in self.text
            stop_word_list = []  # Temp list to append tweet without stop words into self.text
            split_row = row.split()  # Splits the tweet into individual words.
            for word in split_row:  # Loops through each word
                if word not in set(stopwords.words('english')):  # If word not in nltk's stop words list.
                    stop_word_list.append(word)  # Append the word to a temp list.
            tweet[iterator] = ' '.join(
                stop_word_list)  # Turn the list back into a string and replace self.text location.
            iterator += 1
        return tweet

    def print_title(self, string):  # Prints the string to the approx same length
        dash_length = 100 - len(string)
        dashes = '-' * (int(dash_length / 2))
        new_string = (dashes + ' ' + string + ' ' + dashes)
        print(new_string)

    def get_sentiment(self, tweet):
        self.print_title('Performing Sentiment Analysis')

        sentiment_tweet = []
        iterator = 0
        #Following sourced from https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
        for row in tweet:
            tweet_temp = TextBlob(row)
            # sentiment[iterator] = tweet_temp.sentiment.polarity
            sentiment = tweet_temp.sentiment.polarity
            if sentiment == 0.0:

                sentiment_tweet.append((row + ' neutral').replace('\n', ''))
            elif sentiment > 0.0:
                sentiment_tweet.append((row + ' positive').replace('\n', ''))
            else:
                sentiment_tweet.append((row + ' negative').replace('\n', ''))
            iterator += 1
        return sentiment_tweet

