from elasticsearch_dsl import Document, Text, connections, Keyword, Double, Nested, InnerDoc, Float, Integer

connections.create_connection(hosts=['localhost'])

class Sentence(Document):
    text = Text()
    sentiment = Float()
    rating = Float()


class Feature(InnerDoc):
    name = Keyword()
    sentiment = Float()
    positive_sentiment = Float()
    negative_sentiment = Float()
    positive_sentences = Nested(Sentence)
    neutral_sentences = Nested(Sentence)
    negative_sentences = Nested(Sentence)


class Rating(Document):
    avg_rating = Float()
    five_stars = Integer()
    four_stars = Integer()
    three_stars = Integer()
    two_stars = Integer()
    one_stars = Integer()


class Product(Document):
    keywords = Keyword(index=False)
    rating = Nested(Rating)
    features = Nested(Feature)
    reviews = Integer()

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
