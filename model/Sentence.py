from elasticsearch_dsl import Document, Text, connections, Keyword, Double, Nested, InnerDoc, Float

connections.create_connection(hosts=['localhost'])


class Sentence(Document):
    userid = Keyword(index=False)
    productid = Keyword(index=False)
    features = Keyword(index=False)
    text = Text()
    sentiment = Float()
    rating = Float()

    class Index:
        name = 'sentences'
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }

    def save(self, **kwargs):
        return super().save(**kwargs)
