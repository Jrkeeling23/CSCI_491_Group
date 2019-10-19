import pandas as pd
import nltk
from nltk import WordNetLemmatizer, SnowballStemmer, LancasterStemmer
from nltk.corpus import stopwords
import preprocessor as tweet_preprocessor


class Data():
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.text = None  # Text DataFrame
        self.id = None  # ID DataFrame
        self.label = None  # Label DataFrame
        self.get_files()
        self.text = self.text[0:20]  # TODO use full data set
        self.id = self.id[0:20]  # TODO use full data set
        self.label = self.label[0:20]  # TODO use full data set

        self.stop_words()
        self.remove_punctuation()
        self.stem()
        self.lemmatize()
        for i in self.text:
            print(i)

    def get_files(self):  # Get the data files and convert them to pandas DataFrames.
        print(
            '\n----------------------------------------------Getting Data Files----------------------------------------------\n')
        self.text = list(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.text'))  # Makes Text a list
        self.id = pd.DataFrame(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.ids'))
        self.label = pd.DataFrame(open('../data/tweet_by_ID_08_9_2019__04_16_29.txt.labels'))

    def stem(self):  # Method to stem test
        print(
            '\n----------------------------------------------Stemming Data----------------------------------------------\n')
        stemmer = LancasterStemmer()  # Initialize a stemmer
        iterator = 0  # Iterator to replace stemmed sentence back into self.text list
        for row in self.text:  # Loops through each tweet in self.text
            stemmed_list = []  # Temp list to add back into self.text
            split_row = row.split()  # Splits the tweet into words by ' '.
            for word in split_row:  # Loops through each word
                stemmed_list.append(stemmer.stem(word))  # Appends the stemmed word to the temp list.
            self.text[iterator] = ' '.join(
                stemmed_list)  # converts the list to a sentence ands adds back into self.text
            iterator += 1

    def lemmatize(self):  # Method to lemmatize test
        print(
            '\n----------------------------------------------Lemmatizing Data----------------------------------------------\n')
        lemmatizer = WordNetLemmatizer()  # Instantiate a lemmatizer
        iterator = 0
        for row in self.text:  # Loos through each iteration tweet in self.text
            lemmatized_list = []  # Temp list to add back into self.text after tweet is lemmatized
            split_row = row.split()  # Splits the tweet into individual words
            for word in split_row:  # loops through each word.
                lemmatized_list.append(lemmatizer.lemmatize(word))  # Appends the lemmatized word into a temp list.
            self.text[iterator] = ' '.join(lemmatized_list)  # Replaces tweet with a lemmatized tweet.
            iterator += 1

    def remove_punctuation(self):  # Removes tweet punctuation.
        print(
            '\n----------------------------------------------Cleaning Tweets----------------------------------------------\n')
        # Source for tweet cleaning: https://pypi.org/project/tweet-preprocessor/
        iterator = 0
        for row in self.text:  # Loops through each tweet
            self.text[iterator] = tweet_preprocessor.tokenize(row)  # Replaces self.text tweet with the cleaned tweed.
            iterator += 1

    def stop_words(self):  # Removes stop words from tweet.
        # Following stop words code sourced from : https://pythonspot.com/nltk-stop-words/
        print(
            '\n----------------------------------------------Removing Stop Words From Data----------------------------------------------\n')
        iterator = 0
        for row in self.text:  # Loops through each tweet in self.text
            stop_word_list = []  # Temp list to append tweet without stop words into self.text
            split_row = row.split()  # Splits the tweet into individual words.
            for word in split_row:  # Loops through each word
                if word not in set(stopwords.words('english')):  # If word not in nltk's stop words list.
                    stop_word_list.append(word)  # Append the word to a temp list.
            self.text[iterator] = ' '.join(
                stop_word_list)  # Turn the list back into a string and replace self.text location.
            iterator += 1
