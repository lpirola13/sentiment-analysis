import csv
import nltk
import numpy as np
import os
import pandas as pd
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import SO_Run
from model.Product import Product, Feature, Sentence, Rating
from model.Review import Review
from nltk import WordNetLemmatizer, PorterStemmer
from preprocess import Preprocess
from yake import KeywordExtractor

# nltk.download('sentiwordnet')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

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
        num_reviews = 0
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
            text = re.sub(r'\$', ' ', text)
            text = re.sub(r'\"', ' ', text)
            text = re.sub(r'%', ' ', text)
            full_text = full_text + " " + text
            review_object = Review(uderid=review['userid'], productid=review['productid'], text=review['text'],
                                   rating=review['score'])
            review_object.save()
            ratings.append(review['score'])
            num_reviews += 1
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
        product_object = Product(keywords=keywords, rating=rating, reviews=num_reviews)
        product_object.meta.id = product
        product_object.save()


def run_sentiment_analysis_with_SOCAL(dataframe, products, features):
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
                text = review['text']
                text = re.sub(r"http\S+", " ", text)
                text = re.sub(r'<.*?>', ' ', text)
                text = re.sub(r' +', ' ', text)
                text = re.sub(r'\d', ' ', text)
                text = re.sub(r'\.+', '. ', text)
                text = re.sub(r'-', ' ', text)
                text = re.sub(r'\(', ' ', text)
                text = re.sub(r'\)', ' ', text)
                text = re.sub(r'\$', ' ', text)
                text = re.sub(r'\"', ' ', text)
                text = re.sub(r'%', ' ', text)
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


def run_sentiment_analysis_with_VADER(dataframe, products, features):
    sid_obj = SentimentIntensityAnalyzer()
    for product in products:
        print('PRODUCT {}'.format(product))
        features_to_save = []
        for feature in features:
            print('\tFEATURE {}'.format(feature))
            feature_found = False
            sentences = []
            n_saved_files = 0
            for row, review in dataframe[dataframe['productid'] == product].iterrows():
                text = review['text']
                text = re.sub(r"http\S+", " ", text)
                text = re.sub(r'<.*?>', ' ', text)
                text = re.sub(r' +', ' ', text)
                text = re.sub(r'\d', ' ', text)
                text = re.sub(r'\.+', '. ', text)
                text = re.sub(r'-', ' ', text)
                text = re.sub(r'\(', ' ', text)
                text = re.sub(r'\)', ' ', text)
                text = re.sub(r'\$', ' ', text)
                text = re.sub(r'\"', ' ', text)
                text = re.sub(r'%', ' ', text)
                if feature in text:
                    feature_found = True
                    for n_sentence, sentence in enumerate(nltk.sent_tokenize(text)):
                        if feature in sentence:
                            print('\t\tSENTENCE {}'.format(sentence))
                            # Frase che contiene la feature da analizzare
                            sentences.append({"sentence": sentence, "rating": review['score']})
                            n_saved_files += 1
            if feature_found:
                positive_sentences = []
                positive_sentiment = []
                neutral_sentences = []
                negative_sentences = []
                negative_sentiment = []
                sentiment_dataframe = pd.DataFrame(columns=['Sentiment', 'Score', 'Text'])
                print('\tFEATURE {} SENTIMENT ANALYSIS'.format(feature))
                for sentence_dict in sentences:
                    sentence = sentence_dict['sentence']
                    sentiment_dict = sid_obj.polarity_scores(sentence)
                    if sentiment_dict['compound'] >= 0.05:
                        score = round(sentiment_dict['compound'], 2)
                        sentiment = 'positive'
                        positive_sentences.append(
                            Sentence(text=sentence, sentiment=score, rating=sentence_dict['rating']))
                        positive_sentiment.append(score)
                    elif sentiment_dict['compound'] <= - 0.05:
                        score = round(sentiment_dict['compound'], 2)
                        sentiment = 'negative'
                        negative_sentences.append(
                            Sentence(text=sentence, sentiment=score, rating=sentence_dict['rating']))
                        negative_sentiment.append(score)
                    else:
                        score = 0
                        sentiment = 'neutral'
                        neutral_sentences.append(
                            Sentence(text=sentence, sentiment=score, rating=sentence_dict['rating']))
                    sentiment_dataframe = sentiment_dataframe.append(
                        {'Sentiment': sentiment, 'Score': score, 'Text': sentence},
                        ignore_index=True)
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
        # # Aggiorno il prodotto con le varie features e sentiment identificate
        product_object = Product()
        product_object.meta.id = product
        product_object.update(features=features_to_save)
        print('\n\n')


def find_features(dataframe, products):
    noun_score = {}
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
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
            text = re.sub(r'%', ' ', text)
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


dataframe = pd.read_csv(os.path.join(WORKING_DIRECTORY, "assets", "food.csv"), sep=',', index_col=0).reset_index(
    drop=True)
products = dataframe['productid'].unique()

#find_features(dataframe, products)
features = ["flavor", "taste", "product", "price", "quality", "brand", "texture", "package", "variety", "smell"]
persist_products_and_reviews(dataframe, products)
run_sentiment_analysis_with_VADER(dataframe, products, features)
