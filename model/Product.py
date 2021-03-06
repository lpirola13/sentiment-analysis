from elasticsearch_dsl import Document, Text, connections, Keyword, Double, Nested, InnerDoc, Float

connections.create_connection(hosts=['localhost'])


class Feature(InnerDoc):
    name = Keyword()
    sentiment = Float()


class Product(Document):
    keywords = Keyword(index=False)
    rating = Float()
    features = Nested(Feature)

    class Index:
        name = 'products'
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }

    def add_feature(self, name, sentiment):
        self.features.append(Feature(name=name, sentiment=sentiment))

    def save(self, ** kwargs):
        return super().save(** kwargs)
