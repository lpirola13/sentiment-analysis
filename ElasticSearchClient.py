from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import QueryString, Match


class ElasticSearchClient:
    def __init__(self):
        self.es = Elasticsearch(['localhost'])

    def get_products(self, query):
        search = Search(using=self.es)
        search = search[0:5]
        search = search.index('products')
        query = '*{}*'.format(query)
        search = search.filter(QueryString(query=query, fields=['keywords']))
        hits = []
        for hit in search.execute():
            hit_dict = hit.to_dict()
            text = ""
            for index, keyword in enumerate(hit_dict['keywords']):
                if index == 0:
                    text = text + keyword
                else:
                    text = text + " - " + keyword
            hits.append({"label": text, "value": hit.meta.id})
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


if __name__ == '__main__':
    es = ElasticSearchClient()
    es.get_products('sauce')
