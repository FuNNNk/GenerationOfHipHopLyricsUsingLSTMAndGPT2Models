import csv
import glob
import pandas as pd
from better_profanity import profanity
import string
import numpy as np
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
import string

# from gensim.parsing.preprocessing import remove_stopwords


class preprocessLyrics:
    def __init__(self):
        self.csv_file_path = 'convert_sample.csv'
        self.df = pd.read_csv('convert_sample.csv', encoding='UTF8')
        self.ctoi = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11,
                     'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22,
                     'x': 23, 'y': 24, 'z': 25}
        self.itoc = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
                     12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w',
                     23: 'x', 24: 'y', 25: 'z'}

    # adding all extracted verses from Genius to a single csv file
    def addAllVersesCsv(self):
        with open(self.csv_file_path, 'w', encoding='UTF8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['lyrics'])
            for path in glob.glob('./Songs/*.txt'):
                with open(path, encoding='UTF8') as txt_file:
                    txt = txt_file.read() + '\n' + '\t'
                    data = list(txt.split('\n\t'))
                    writer.writerow([data[0]])

    # Removing the profane words present in lyrics
    def removeCurseWords(self):
        custom_badwords = ['motherfuckin\'', 'hoes', 'hoe', 'sumbitch', 'Faggot', 'Nigga', 'motherfuckas', 'fuckin\'',
                           'sucker', 'Niggas\'ll', 'ma\'fucker']
        profanity.add_censor_words(custom_badwords)

        for i, row in self.df.iterrows():
            censoredRow = profanity.censor(row['lyrics'], '')
            censoredRowLower = censoredRow.lower()
            censoredRowNoPunc = censoredRowLower.translate(str.maketrans('', '', string.punctuation))
            if i % 10 == 0:
                print(i)
            self.df.at[i, 'lyrics'] = censoredRowNoPunc
        self.df.to_csv(self.csv_file_path, index=False)

    # adding a sentiment column to the DataFrame of verses
    def addSentiment(self):
        self.df = pd.read_csv(self.csv_file_path, encoding='UTF8')

        with open('sentiment.txt', 'r', encoding='UTF8') as txt_file:
            new_sentiment_column = txt_file.read()
            list_of_sentiments = new_sentiment_column.replace("[", "").replace('(', '').replace("'", '').replace(']',
                                                                                                                 '').replace(
                '\n', '').split('), ')

        self.df['sentiment'] = list_of_sentiments
        for i, row in self.df.iterrows():
            if not row['sentiment']:
                self.df.drop([i], inplace=True)
        self.df.to_csv(self.csv_file_path, index=False)

    # def characterToInteger(self):
    #     self.df = pd.read_csv(self.csv_file_path, encoding='UTF8')
    #     transformedDF = pd.DataFrame({'lyrics': []})
    #     for i, row in self.df.iterrows():
    #         transformedList = []
    #         for char in row['lyrics']:
    #             if re.search('[a-z]', char) is not None:
    #                 transformedList.append(self.ctoi[char.lower()])
    #         # print(transformedList)
    #         transformedTemporaryDf = pd.DataFrame({'lyrics': [transformedList]})
    #         transformedDF = transformedDF._append(transformedTemporaryDf)
    #     transformedDF.to_csv('transformedDF.csv', index=False)


    def tokenizeCorpus(self, corpus):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(corpus)
        return tokenizer

    def lyricsCorpus(self):
        lyrics = self.df['lyrics'].str.cat()
        corpus = lyrics.split('\n')
        for item in range(len(corpus)):
            corpus[item] = corpus[item].rstrip()
        corpus = [item for item in corpus if item!='']
        return corpus

    def training(self):
        corpus = self.lyricsCorpus()
        tokenizer = self.tokenizeCorpus(corpus)
        vocab_size = len(tokenizer.word_index)+1

        seq = []
        for item in corpus:
            seq_list = tokenizer.texts_to_sequences([item])[0]
            for i in range(1,len(seq_list)):
                n_gram = seq_list[:i+1]
                seq.append(n_gram)

        max_seq_size = max([len(s) for s in seq])
        seq = np.array(pad_sequences(seq,maxlen=max_seq_size,padding='pre'))

        input_sequences, labels = seq[:,:-1], seq[:,-1]
        one_hot_labels = to_categorical(labels, num_classes=vocab_size)

        model=Sequential()
        model.add(Embedding(vocab_size,64,input_length=max_seq_size-1))
        model.add(Bidirectional(LSTM(20)))
        model.add(Dense(vocab_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(input_sequences, one_hot_labels, epochs = 100, verbose=1)

        model.save('generativeModel.h5')


#### Continue with reading this https://arxiv.org/pdf/2004.03965



    # def integerToCharacter(self):
    #     self.df = pd.read_csv(self.csv_file_path, encoding='UTF8')
    #     for i, row in self.df.iterrows():
    #         self.itoc[row['character']] = i

    # def removeStopWords(self):
    #     for i, row in self.df.iterrows():
    #         if i % 10 == 0:
    #             print(i)
    #         self.df.at[i, 'lyrics'] = remove_stopwords(row['lyrics'])
    #     self.df.to_csv(self.csv_file_path, index=False)
