from elasticsearch_dsl import Document, Text, connections, Double, Keyword, Float

connections.create_connection(hosts=['localhost'])


class Review(Document):
    uderid = Keyword(index=False)
    productid = Keyword(index=False)
    text = Text()
    rating = Float()

    class Index:
        name = 'reviews'
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }

    def save(self, **kwargs):
        return super().save(**kwargs)
