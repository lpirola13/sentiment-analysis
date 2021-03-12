from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import QueryString, Match, Bool, MatchAll
import pandas as pd


def make_column_names():
    columns = ['productid']
    features = ["flavor", "taste", "product", "price", "quality", "brand", "texture", "package", "variety", "smell"]
    for feature in features:
        columns.extend([
            "{}_n_pos".format(feature),
            "{}_avg_pos".format(feature),
            "{}_sum_pos".format(feature),
            "{}_n_neg".format(feature),
            "{}_abg_neg".format(feature),
            "{}_sum_neg".format(feature),
            "{}_n_neu".format(feature),
            "{}_avg".format(feature),
        ])
    return columns


def get_labels(sentiment, rating):
    if sentiment > 0:
        prediction = 'positive'
    elif sentiment < 0:
        prediction = 'negative'
    else:
        prediction = 'neutral'
    if rating > 3:
        real = 'positive'
    elif rating < 3:
        real = 'negative'
    else:
        real = 'neutral'
    return prediction, real


class ElasticSearchClient:
    def __init__(self):
        self.es = Elasticsearch(['localhost'])

    def get_products(self, query):
        search = Search(using=self.es)
        search = search.index('products')
        search = search.query(
            Bool(should=[Match(_id=query), QueryString(query='{}*'.format(query), fields=['keywords'])])).sort(
            '-reviews')
        hits = []
        for hit in search.execute():
            hit_dict = hit.to_dict()
            text = ""
            for index, keyword in enumerate(hit_dict['keywords']):
                if index == 0:
                    text = text + keyword
                else:
                    text = text + " - " + keyword
            if len(text) > 50:
                hits.append({"label": '{}: {}...'.format(hit.meta.id, text[0:50]), "value": hit.meta.id})
            else:
                hits.append({"label": '{}: {}'.format(hit.meta.id, text), "value": hit.meta.id})
        return hits

    def get_product(self, id):
        search = Search(using=self.es)
        search = search.index('products')
        search = search.query(Match(_id=id))
        response = search.execute()
        product = None
        if len(response) > 0:
            product = response
        return product

    def get_statistics(self):
        dataframe = pd.DataFrame(columns=['feature', 'label', 'prediction', 'text'])
        search = Search(using=self.es)
        search = search.index('products')
        search = search.source(['features.positive_sentences.text', 'features.positive_sentences.rating',
                                'features.positive_sentences.sentiment', 'features.name'])
        search = search.query(MatchAll())
        print('positive')
        i = 0
        for product in search.scan():
            print(i)
            product_dict = product.to_dict()
            if 'features' in product_dict:
                for feature in product_dict['features']:
                    if 'positive_sentences' in feature:
                        for sentence in feature['positive_sentences']:
                            prediction, real = get_labels(sentence['sentiment'], int(sentence['rating']))
                            dataframe.loc[len(dataframe)] = [feature['name'], real, sentence['sentiment'],
                                                             sentence['text']]
            i += 1
        search = search.index('products')
        search = search.source(['features.neutral_sentences.text', 'features.neutral_sentences.rating',
                                'features.neutral_sentences.sentiment', 'features.name'])
        search = search.query(MatchAll())
        print('neutral')
        i = 0
        for product in search.scan():
            product_dict = product.to_dict()
            print(i)
            if 'features' in product_dict:
                for feature in product_dict['features']:
                    if 'neutral_sentences' in feature:
                        for sentence in feature['neutral_sentences']:
                            prediction, real = get_labels(sentence['sentiment'], int(sentence['rating']))
                            dataframe.loc[len(dataframe)] = [feature['name'], real, sentence['sentiment'],
                                                             sentence['text']]
            i += 1
        search = search.index('products')
        search = search.source(['features.negative_sentences.text', 'features.negative_sentences.rating',
                                'features.negative_sentences.sentiment', 'features.name'])
        search = search.query(MatchAll())
        print('negative')
        i = 0
        for product in search.scan():
            product_dict = product.to_dict()
            print(i)
            if 'features' in product_dict:
                for feature in product_dict['features']:
                    if 'negative_sentences' in feature:
                        for sentence in feature['negative_sentences']:
                            prediction, real = get_labels(sentence['sentiment'], int(sentence['rating']))
                            dataframe.loc[len(dataframe)] = [feature['name'], real, sentence['sentiment'],
                                                             sentence['text']]
            i += 1
        # dataframe = dataframe.drop_duplicates(subset=['text', 'label']).reset_index(drop=True)
        return dataframe


if __name__ == '__main__':
    es = ElasticSearchClient()
    es.get_products('sauce')
