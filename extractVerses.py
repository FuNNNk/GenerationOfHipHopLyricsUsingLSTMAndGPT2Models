import csv
import glob
import pandas as pd

def addAllVersesCsv():
    with open('convert_sample.csv', 'w', encoding='UTF8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['lyrics'])
        for path in glob.glob('./Songs/*.txt'):
            with open(path, encoding='UTF8') as txt_file:
                txt = txt_file.read() + '\n' + '\t'
                data = list(txt.split('\n\t'))
                writer.writerow([data[0]])

def addSentiment():
    csv_file_path = 'convert_sample.csv'
    df = pd.read_csv(csv_file_path, encoding='UTF8')

    with open ('sentiment.txt', 'r', encoding='UTF8') as txt_file:
        new_sentiment_column = txt_file.read()
        list_of_sentiments = new_sentiment_column.replace("[", "").replace('(','').replace("'",'').replace(']','').replace('\n','').split('), ')

    df['sentiment'] = list_of_sentiments
    df.to_csv(csv_file_path, index=False)

# addAllVersesCsv()
# addSentiment()

