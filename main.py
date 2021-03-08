# import collections
# import csv
import csv
import os
import re
# from math import floor
# from pathlib import Path
#
# import gensim as gensim
# import lda as lda
# import nltk
from math import floor

import nltk
import pandas as pd
import numpy as np
# import spacy
# import pyfpgrowth
# from bs4 import BeautifulSoup
# from efficient_apriori import apriori
# from keybert import KeyBERT
# from pyarc import CBA, TransactionDB
# from pyarc.algorithms import top_rules, createCARs, generateCARs
# from mlxtend.preprocessing import TransactionEncoder
# from nltk.corpus import sentiwordnet as swn, stopwords
# from senticnet.senticnet import SenticNet

# nltk.download('sentiwordnet')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# from sklearn.feature_extraction.text import CountVectorizer
#
# import SO_Run
# from preprocess import Preprocess

# nlp = spacy.load("en_core_web_sm")
# # nlp.add_pipe('merge_noun_chunks')
#
from nltk import WordNetLemmatizer, PorterStemmer

import SO_Run
from model.Product import Product, Feature, Sentence, Rating
from model.Review import Review

from yake import KeywordExtractor
from rake_nltk import Rake

from preprocess import Preprocess

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

WORKING_DIRECTORY = os.getcwd()


def extract_keywords(text):
    print(text)
    extractor = KeywordExtractor(lan="en", top=3, windowsSize=3)
    keywords = extractor.extract_keywords(text=text)
    return [x for x, y in keywords]


def persist_products_and_reviews(dataframe, products):
    for index_product, product in enumerate(products, start=1):
        print('PRODUCT {} {} of {}'.format(product, index_product, len(products)))
        full_text = ""
        ratings = []
        for row, review in dataframe[dataframe['productid'] == product].iterrows():
            text = review['text'].lower()
            text = re.sub(r"http\S+", " ", text)
            text = re.sub(r'<.*?>', ' ', text)
            text = re.sub(r' +', ' ', text)
            text = re.sub(r'\d', ' ', text)
            text = re.sub(r'\.+', '. ', text)
            text = re.sub(r'-', ' ', text)
            text = re.sub(r'!', ' ', text)
            text = re.sub(r'\(', ' ', text)
            text = re.sub(r'\)', ' ', text)
            full_text = full_text + " " + text
            review_object = Review(uderid=review['userid'], productid=review['productid'], text=review['text'],
                                   rating=review['score'])
            review_object.save()
            ratings.append(review['score'])
        keywords = extract_keywords(full_text)
        print('\tKEYWORDS {}'.format(keywords))
        avg_rating = np.mean(ratings)
        five_stars = ratings.count(5)
        four_stars = ratings.count(4)
        three_stars = ratings.count(3)
        two_stars = ratings.count(2)
        one_stars = ratings.count(1)
        rating = Rating(avg_rating=avg_rating,
                        five_stars=five_stars,
                        four_stars=four_stars,
                        three_stars=three_stars,
                        two_stars=two_stars,
                        one_stars=one_stars)
        product_object = Product(keywords=keywords, rating=rating)
        product_object.meta.id = product
        product_object.save()


def run_sentiment_analysis(dataframe, products, features):
    # Creo cartella input
    input_path = os.path.join(WORKING_DIRECTORY, "input")
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    # Creo cartella output
    output_path = os.path.join(WORKING_DIRECTORY, "output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # Creo cartella sentiment
    sentiment_path = os.path.join(WORKING_DIRECTORY, "sentiment")
    if not os.path.exists(sentiment_path):
        os.mkdir(sentiment_path)

    config_file = os.path.join(WORKING_DIRECTORY, "Resources", "config_files", "en_SO_Calc.ini")

    for product in products:
        print('PRODUCT {}'.format(product))
        # Creo la cartella del prodotto in input
        product_input_path = os.path.join(input_path, product)
        product_output_path = os.path.join(output_path, product)
        if not os.path.exists(product_input_path):
            os.mkdir(product_input_path)
        # Creo la cartella del prodotto in sentiment
        product_sentiment_path = os.path.join(sentiment_path, product)
        if not os.path.exists(product_sentiment_path):
            os.mkdir(product_sentiment_path)
        features_to_save = []
        for feature in features:
            print('\tFEATURE {}'.format(feature))
            feature_found = False
            feature_input_path = ''
            feature_output_path = ''
            sentiment_output_path = ''
            n_saved_files = 0
            for row, review in dataframe[dataframe['productid'] == product].iterrows():
                text = review['text'].lower()
                text = re.sub(r"http\S+", " ", text)
                text = re.sub(r'<.*?>', ' ', text)
                text = re.sub(r' +', ' ', text)
                text = re.sub(r'\d', ' ', text)
                text = re.sub(r'\.+', '. ', text)
                text = re.sub(r'-', ' ', text)
                text = re.sub(r'!', ' ', text)
                text = re.sub(r'\(', ' ', text)
                text = re.sub(r'\)', ' ', text)
                if feature in text:
                    feature_found = True
                    # Creo la cartella della feature nella cartella del prodotto in input
                    feature_input_path = os.path.join(product_input_path, feature)
                    feature_output_path = os.path.join(product_output_path, feature)
                    sentiment_output_path = os.path.join(product_sentiment_path, feature)
                    if not os.path.exists(feature_input_path):
                        os.mkdir(feature_input_path)
                    for n_sentence, sentence in enumerate(nltk.sent_tokenize(text)):
                        if feature in sentence:
                            print('\t\tSENTENCE {}'.format(sentence))
                            # Salvo la frase per poter eseguire il preprocessing
                            filename = str(n_saved_files) + '-' + review['userid'] + '-' + str(n_sentence) + '-' + str(
                                int(review['score'])) + '.txt'
                            sentence_input_path = os.path.join(feature_input_path, filename)
                            with open(sentence_input_path, 'w') as file:
                                file.write(sentence)
                            n_saved_files += 1
            if feature_found:
                positive_sentences = []
                positive_sentiment = []
                neutral_sentences = []
                negative_sentences = []
                negative_sentiment = []
                print('\tFEATURE {} PREPROCESSING'.format(feature))
                p = Preprocess(feature_input_path, feature_output_path, 'tokenize,ssplit,pos')
                p.pos_tagging()
                print('\tFEATURE {} SENTIMENT ANALYSIS'.format(feature))
                SO_Run.main(feature_output_path, sentiment_output_path, config_file, 0.0)
                sentiment_file = os.path.join(sentiment_output_path, "file_sentiment.csv")
                sentiment_dataframe = pd.read_csv(sentiment_file, header=0, sep=',')
                sentiment_dataframe = sentiment_dataframe.sort_values('File_Name')
                print('\tFEATURE {} RESULTS'.format(feature))
                print('\t\tNUMBER OF REVIEWS {}'.format(len(sentiment_dataframe.index)))
                positive = len(sentiment_dataframe[sentiment_dataframe['Sentiment'] == 'positive'].index)
                neutral = len(sentiment_dataframe[sentiment_dataframe['Sentiment'] == 'neutral'].index)
                negative = len(sentiment_dataframe[sentiment_dataframe['Sentiment'] == 'negative'].index)
                print('\t\tNUMBER OF POSITIVE REVIEWS {}'.format(positive))
                print('\t\tNUMBER OF NEUTRAL REVIEWS {}'.format(neutral))
                print('\t\tNUMBER OF NEGATIVE REVIEWS {}'.format(negative))
                overall_score = sentiment_dataframe['Score'].mean()
                if overall_score > 0:
                    sentiment = 'positive'
                elif overall_score < 0:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                print('\t\tOVERALL SENTIMENT {} {}'.format(sentiment, round(overall_score, 2)))
                for number, result in sentiment_dataframe.iterrows():
                    file_path = os.path.join(feature_input_path, result['File_Name'])
                    rating = str(file_path)[len(file_path) - 5:len(file_path) - 4]
                    with open(file_path) as f:
                        for line in f:
                            print('\t\t\t' + str(round(result['Score'], 2)) + ' - ' + line)
                    score = round(result['Score'], 2)
                    if score > 0:
                        positive_sentences.append(Sentence(text=line, sentiment=score, rating=rating))
                        positive_sentiment.append(score)
                    elif score < 0:
                        negative_sentences.append(Sentence(text=line, sentiment=score, rating=rating))
                        negative_sentiment.append(score)
                    else:
                        neutral_sentences.append(Sentence(text=line, sentiment=score, rating=rating))
                print('\n')
                if len(positive_sentiment) > 0:
                    avg_positive_sentiment = round(np.mean(positive_sentiment), 2)
                else:
                    avg_positive_sentiment = 0.0
                if len(negative_sentiment) > 0:
                    avg_negative_sentiment = round(np.mean(negative_sentiment), 2)
                else:
                    avg_negative_sentiment = 0.0
                feature_to_save = Feature(name=feature,
                                          sentiment=round(overall_score, 2),
                                          positive_sentiment=avg_positive_sentiment,
                                          negative_sentiment=avg_negative_sentiment,
                                          positive_sentences=positive_sentences,
                                          neutral_sentences=neutral_sentences,
                                          negative_sentences=negative_sentences)
                features_to_save.append(feature_to_save)
            else:
                print('\t\tNULL')
                feature_to_save = Feature(name=feature,
                                          sentiment=None,
                                          positive_sentiment=0.0,
                                          negative_sentiment=0.0,
                                          positive_sentences=[Sentence()],
                                          neutral_sentences=[Sentence()],
                                          negative_sentences=[Sentence()])
                features_to_save.append(feature_to_save)
        # Aggiorno il prodotto con le varie features e sentiment identificate
        product_object = Product()
        product_object.meta.id = product
        product_object.update(features=features_to_save)
        print('\n\n')


def find_features(dataframe, products):
    noun_score = {}
    lemmatizer = WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    for product in products:
        for row, review in dataframe[dataframe['productid'] == product].iterrows():
            print('ORIGINAL: {}'.format(review['text']))
            text = review['text'].lower()
            text = re.sub(r"http\S+", " ", text)
            text = re.sub(r'<.*?>', ' ', text)
            text = re.sub(r' +', ' ', text)
            text = re.sub(r'\d', ' ', text)
            text = re.sub(r'\.+', '. ', text)
            text = re.sub(r'-', ' ', text)
            text = re.sub(r'!', ' ', text)
            text = re.sub(r'\(', ' ', text)
            text = re.sub(r'\)', ' ', text)
            for sentence in nltk.sent_tokenize(text):
                print('\tSENTENCE: {}'.format(sentence))
                tokens = nltk.word_tokenize(sentence)
                tokens = (nltk.pos_tag(tokens))
                phrase = [(lemmatizer.lemmatize(element[0]), element[1]) for element in tokens if
                          not element[0] in stopwords]
                for index, element in enumerate(phrase):
                    if element[1] == 'JJ':
                        for i in range(1, len(phrase)):
                            if index - i > 0:
                                if phrase[index - i][1] == 'NN':
                                    if phrase[index - i][0] in noun_score:
                                        noun_score[phrase[index - i][0]] += 1
                                    else:
                                        noun_score[phrase[index - i][0]] = 0
                                    break
                            if index + i < len(phrase):
                                if phrase[index + i][1] == 'NN':
                                    if phrase[index + i][0] in noun_score:
                                        noun_score[phrase[index + i][0]] += 1
                                    else:
                                        noun_score[phrase[index + i][0]] = 0
                                    break
            print('\n')

    noun_score = dict(sorted(noun_score.items(), key=lambda item: item[1], reverse=True))
    print('NOUNS FOUND {}'.format(len(noun_score)))

    with open('features.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in noun_score.items():
            writer.writerow([key, value])


dataframe = pd.read_csv(os.path.join(WORKING_DIRECTORY, "assets", "food.tsv"), sep='\t')

reviews = pd.DataFrame(dataframe['productid'].value_counts())
print(len(dataframe))
print(dataframe['productid'].nunique())
print(dataframe['userid'].nunique())
dataframe = dataframe[0:50]

products = dataframe['productid'].unique()
print(products)

# find_features(dataframe, products)

# persist_products_and_reviews(dataframe, products)
features = ['flavor', 'taste', 'brand', 'quality', 'price', 'product']
run_sentiment_analysis(dataframe, products, features)

# product_object = Product()
# product_object.meta.id = 'B001E4KFG0'
# product_object.update(features=[Feature(name='flavor', sentiment=0.5), Feature(name='taste', sentiment=0.5)])


# final_nouns = []
# final_adjectives = []
# noun_score = {}
#
# for product in products:
#     for row, review in dataframe[dataframe['productid'] == product].iterrows():
#         print('ORIGINAL: {}'.format(review['text']))
#         text = review['text'].lower()
#         text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
#         text = re.sub(r'\<a href', ' ', text)
#         text = re.sub(r'&amp;', '', text)
#         text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
#         text = re.sub(r'<br />', ' ', text)
#         text = re.sub(r'\'', ' ', text)
#         doc = nlp(text)
#         for sentence in doc.sents:
#             print('\tSENTENCE: {}'.format(sentence))
#             sen = nlp(str(sentence))
#             phrase = [(token.lemma_, token.pos_) for token in sen]
#             for index, element in enumerate(phrase):
#                 if element[1] == 'ADJ':
#                     for i in range(1, len(phrase)):
#                         if index - i > 0:
#                             if phrase[index - i][1] == 'NOUN':
#                                 if phrase[index - i][0] in noun_score:
#                                     noun_score[phrase[index - i][0]] += 1
#                                 else:
#                                     noun_score[phrase[index - i][0]] = 0
#                                 break
#                         if index + i < len(phrase):
#                             if phrase[index + i][1] == 'NOUN':
#                                 if phrase[index + i][0] in noun_score:
#                                     noun_score[phrase[index + i][0]] += 1
#                                 else:
#                                     noun_score[phrase[index + i][0]] = 0
#                                 break
#         print('\n')

# print(noun_score)
# noun_score = dict(sorted(noun_score.items(), key=lambda item: item[1], reverse=True))
# print(len(noun_score))
# threshold = sum(noun_score.values()) * 0.5
# threshold = floor(threshold)
# print(threshold)
# sum_value = 0
# for k, v in list(noun_score.items()):
#     if sum_value > threshold:
#         del noun_score[k]
#     else:
#         sum_value = sum_value + v
#
# with open('features.csv', 'w') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(list(noun_score.keys()))


# features = ['flavor', 'taste', 'brand', 'quality', 'price', 'product']
# sn = SenticNet()
#
# for row, review in dataframe[dataframe['productid'] == 'B000E4AKQ6'].iterrows():
#     print('ORIGINAL: {}'.format(review['text']))
#     text = review['text'].lower()
#     text = re.sub(r"http\S+", " ", text)
#     text = BeautifulSoup(text, 'lxml').get_text()
#     text = re.sub(r'<.*?>', ' ', text)
#     print('PREPROCESSED: {}'.format(text))
#     doc = nlp(text)
#     for sentence in doc.sents:
#         sen = nlp(str(sentence))
#         sen_tokens = [token.lemma_ for token in sen if not token.is_punct and not token.is_space]
#         sentence = " ".join(sen_tokens)
#         sentence = re.sub(' +', ' ', sentence)
#         print('\tSENTENCE: {}'.format(sentence))
#         sen = nlp(str(sentence))
#         for feature in features:
#             if feature in str(sentence):
#                 tokens = []
#                 tokens_pos = []
#                 for token in sen:
#                     if not token.is_punct:
#                         tokens_pos.append((token.lemma_, token.pos_))
#                         tokens.append(token.lemma_)
#                 print(tokens_pos)
#                 for i in range(0, len(tokens)):
#                     if tokens_pos[i][1] == 'ADJ':
#                         adj = tokens_pos[i][0]
#                         try:
#                             print(feature, adj)
#                             polarity_label = sn.polarity_label(adj)
#                             polarity_value = sn.polarity_value(adj)
#                             print(polarity_label, polarity_value)
#                         except KeyError:
#                             pass
#
#     print('\n')

#
# features = ['flavor', 'taste', 'brand', 'quality', 'price', 'product']
# sn = SenticNet()
#
# for row, review in dataframe[dataframe['productid'] == 'B0006VB3TA'].iterrows():
#     print('ORIGINAL: {}'.format(review['text']))
#     text = review['text'].lower()
#     text = re.sub(r"http\S+", " ", text)
#     text = re.sub(r'<.*?>', ' ', text)
#     print('PREPROCESSED: {}'.format(text))
#     sentences = nltk.sent_tokenize(text)
#     for sentence in sentences:
#         print(sentence)
#         tokens = nltk.word_tokenize(sentence)
#         tags = nltk.pos_tag(tokens)
#         words = []
#         for word in tokens:
#             if word == 'n\'t':
#                 words.append('not')
#             elif word.isalpha():
#                 words.append(word)
#         from nltk.corpus import stopwords
#         from nltk.stem import WordNetLemmatizer
#         lemmatizer = WordNetLemmatizer()
#         stop_words = set(stopwords.words('english'))
#         final = []
#         for w in words:
#             if w == 'but' or w == 'not':
#                 final.append(w)
#             elif not w in stop_words:
#                 final.append(w)
#         print(final)
#         for feature in features:
#             if feature in final:
#                 adjectives = [adj for adj, pos in tags if pos == 'JJ']
#                 adj_polarities = []
#                 for adj in adjectives:
#                     try:
#                         polarity_label = sn.polarity_label(adj)
#                         polarity_value = sn.polarity_value(adj)
#                         adj_polarities.append((adj, polarity_label))
#                     except KeyError:
#                         pass
#                 print(feature, adj_polarities)
#         sen = nlp(str(sentence))
#         sen_tokens = [token.lemma_ for token in sen if not token.is_punct and not token.is_space]
#         sentence = " ".join(sen_tokens)
#         sentence = re.sub(' +', ' ', sentence)
#         print('\tSENTENCE: {}'.format(sentence))
#         sen = nlp(str(sentence))
#         for feature in features:
#             if feature in str(sentence):
#                 tokens = []
#                 tokens_pos = []
#                 for token in sen:
#                     if not token.is_punct:
#                         tokens_pos.append((token.lemma_, token.pos_))
#                         tokens.append(token.lemma_)
#                 print(tokens_pos)
#                 for i in range(0, len(tokens)):
#                     if tokens_pos[i][1] == 'ADJ':
#                         adj = tokens_pos[i][0]
#                         try:
#                             print(feature, adj)
#                             polarity_label = sn.polarity_label(adj)
#                             polarity_value = sn.polarity_value(adj)
#                             print(polarity_label, polarity_value)
#                         except KeyError:
#                             pass


# products = []
#
# dirname = os.path.dirname(os.path.abspath(__file__))
# if not os.path.exists(dirname + "/input/"):
#     os.mkdir(dirname + "/input/")
# if not os.path.exists(dirname + "/output/"):
#     os.mkdir(dirname + "/output/")
# if not os.path.exists(dirname + "/sentiment/"):
#     os.mkdir(dirname + "/sentiment/")
# for product in products:
#     print('PRODUCT {}'.format(product))
#     if not os.path.exists(dirname + "/input/" + product):
#         os.mkdir(dirname + "/input/" + product)
#     if not os.path.exists(dirname + "/sentiment/" + product):
#         os.mkdir(dirname + "/sentiment/" + product)
#     for row, review in dataframe[dataframe['productid'] == product].iterrows():
#         print('USER {}'.format(review['userid']))
#         if not os.path.exists(dirname + "/input/" + product + '/' + review['userid']):
#             os.mkdir(dirname + "/input/" + product + '/' + review['userid'])
#         if not os.path.exists(dirname + "/sentiment/" + product + '/' + review['userid']):
#             os.mkdir(dirname + "/sentiment/" + product + '/' + review['userid'])
#         text = review['text'].lower()
#         text = re.sub(r"http\S+", " ", text)
#         text = re.sub(r'<.*?>', ' ', text)
#         text = re.sub(r' +', ' ', text)
#         doc = nlp(text)
#         print('\tSENTENCES')
#         for index, sentence in enumerate(doc.sents):
#             print('\t\tSENTENCE {}'.format(str(sentence)))
#             input_path = dirname + "/input/" + product + '/' + review['userid'] + '/' + str(index) + '.txt'
#             with open(input_path, 'w') as file:
#                 file.write(str(sentence))
#         print('\tPREPROCESSING')
#         input_path = dirname + "/input/" + product + '/' + review['userid']
#         output_path = dirname + "/output/" + product + '/' + review['userid']
#         p = Preprocess(input_path, output_path, 'tokenize,ssplit,pos')
#         p.pos_tagging()
#         print('\tSENTIMENT ANALYSIS')
#         sentiment_output_path = dirname + "/sentiment/" + product + '/' + review['userid']
#         config_file = '/Users/lorenzopirola/GitHub/pythonProject/Resources/config_files/en_SO_Calc.ini'
#         SO_Run.main(output_path, sentiment_output_path, config_file, 0.0)
#         sentiment_dataframe = pd.read_csv(sentiment_output_path + '/file_sentiment.csv', header=0, sep=',')
#         sentiment_dataframe = sentiment_dataframe.sort_values('File_Name')
#         print('\tRESULTS')
#         for number, result in sentiment_dataframe.iterrows():
#             print('\t\t' + result['File_Name'] + ' ' + result['Sentiment'])
#         print('\n')
#     print('\n\n')

# # SENTIMENT ANALYSIS UFFICIALE
#
# dataframe = dataframe[0:50]
# # Lista di prodotti da analizzare
# products = dataframe['productid'].unique()
#
# # Lista di features dei prodotti da analizzare
# features = ['flavor', 'taste', 'brand', 'quality', 'price', 'product']
#
# dirname = os.path.dirname(os.path.abspath(__file__))
#
# # Creo cartella input
# input_path = dirname + "/input/"
# if not os.path.exists(input_path):
#     os.mkdir(input_path)
# # Creo cartella output
# output_path = dirname + "/output/"
# if not os.path.exists(output_path):
#     os.mkdir(output_path)
# # Creo cartella sentiment
# sentiment_path = dirname + "/sentiment/"
# if not os.path.exists(sentiment_path):
#     os.mkdir(sentiment_path)
#
# for product in products:
#     print('PRODUCT {}'.format(product))
#     # Creo la cartella del prodotto in input
#     product_input_path = input_path + product
#     product_output_path = output_path + product
#     if not os.path.exists(product_input_path):
#         os.mkdir(product_input_path)
#     # Creo la cartella del prodotto in sentiment
#     product_sentiment_path = sentiment_path + product
#     if not os.path.exists(product_sentiment_path):
#         os.mkdir(product_sentiment_path)
#     for feature in features:
#         print('\tFEATURE {}'.format(feature))
#         feature_found = False
#         feature_input_path = ''
#         feature_output_path = ''
#         sentiment_output_path = ''
#         n_saved_files = 0
#         for row, review in dataframe[dataframe['productid'] == product].iterrows():
#             text = review['text'].lower()
#             text = re.sub(r"http\S+", " ", text)
#             text = re.sub(r'<.*?>', ' ', text)
#             text = re.sub(r' +', ' ', text)
#             if feature in text:
#                 feature_found = True
#                 # Creo la cartella della feature nella cartella del prodotto in input
#                 feature_input_path = product_input_path + '/' + feature
#                 feature_output_path = product_output_path + '/' + feature
#                 sentiment_output_path = product_sentiment_path + '/' + feature
#                 if not os.path.exists(feature_input_path):
#                     os.mkdir(feature_input_path)
#                 for n_sentence, sentence in enumerate(nltk.sent_tokenize(text)):
#                     if feature in sentence:
#                         print('\t\tSENTENCE {}'.format(sentence))
#                         # Salvo la frase per poter eseguire il preprocessing
#                         sentence_input_path = feature_input_path + '/' + str(n_saved_files) + '-' + review['userid'] + '-' + str(n_sentence) + '.txt'
#                         with open(sentence_input_path, 'w') as file:
#                             file.write(sentence)
#                         n_saved_files += 1
#         if feature_found:
#             print('\tFEATURE {} PREPROCESSING'.format(feature))
#             p = Preprocess(feature_input_path, feature_output_path, 'tokenize,ssplit,pos')
#             p.pos_tagging()
#             print('\tFEATURE {} SENTIMENT ANALYSIS'.format(feature))
#             config_file = '/Users/lorenzopirola/GitHub/pythonProject/Resources/config_files/en_SO_Calc.ini'
#             SO_Run.main(feature_output_path, sentiment_output_path, config_file, 0.0)
#             sentiment_dataframe = pd.read_csv(sentiment_output_path + '/file_sentiment.csv', header=0, sep=',')
#             sentiment_dataframe = sentiment_dataframe.sort_values('File_Name')
#             print('\tFEATURE {} RESULTS'.format(feature))
#             print('\tNUMBER OF REVIEWS {}'.format(len(sentiment_dataframe.index)))
#             positive = len(sentiment_dataframe[sentiment_dataframe['Sentiment'] == 'positive'].index)
#             neutral = len(sentiment_dataframe[sentiment_dataframe['Sentiment'] == 'neutral'].index)
#             negative = len(sentiment_dataframe[sentiment_dataframe['Sentiment'] == 'negative'].index)
#             print('\tNUMBER OF POSITIVE REVIEWS {}'.format(positive))
#             print('\tNUMBER OF NEUTRAL REVIEWS {}'.format(neutral))
#             print('\tNUMBER OF NEGATIVE REVIEWS {}'.format(negative))
#             overall_score = sentiment_dataframe['Score'].mean()
#             if overall_score > 0:
#                 sentiment = 'positive'
#             elif overall_score < 0:
#                 sentiment = 'negative'
#             else:
#                 sentiment = 'neutral'
#             print('\tOVERALL SENTIMENT {} {}'.format(sentiment, round(overall_score,2)))
#             for number, result in sentiment_dataframe.iterrows():
#                 print('\t\t' + str(number) + ' - ' + result['Sentiment'])
#             print('\n')
#     print('\n\n')


#
# from yake import KeywordExtractor
# from rake_nltk import Rake
#
#
# dataframe = dataframe[0:100]
# # Lista di prodotti da analizzare
# products = dataframe['productid'].unique()
#
# for product in products:
#     print('PRODUCT {}'.format(product))
#     full_text = ""
#     for row, review in dataframe[dataframe['productid'] == product].iterrows():
#         text = review['text'].lower()
#         text = re.sub(r"http\S+", " ", text)
#         text = re.sub(r'<.*?>', ' ', text)
#         text = re.sub(r' +', ' ', text)
#         text = re.sub(r'\d', ' ', text)
#         text = re.sub(r'\.+', '.', text)
#         text = re.sub(r'\-', ' ', text)
#         text = re.sub(r'\!', ' ', text)
#         text = re.sub(r'\(', ' ', text)
#         text = re.sub(r'\)', ' ', text)
#         full_text = full_text + " " + text
#         #full_text.append(text)
#     kw_extractor = KeywordExtractor(lan="en", n=1, top=3)
#     keywords = kw_extractor.extract_keywords(text=full_text)
#     keywords = [x for x, y in keywords]
#     print('\tKEYWORDS {}'.format(keywords))
#     r = Rake()
#     r.extract_keywords_from_text(full_text)
#     print('\tLONGER DESCRIPTION {}'.format(r.get_ranked_phrases()[0:3]))
#
#
