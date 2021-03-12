import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ElasticSearchClient import ElasticSearchClient

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 3000)

plt.style.use('ggplot')

WORKING_DIRECTORY = os.getcwd()


def preprocessing():
    dataframe = pd.read_csv(os.path.join(WORKING_DIRECTORY, "assets", "food.tsv"), sep='\t')
    print("Total reviews {}".format(len(dataframe)))
    print("Number of products {}".format(dataframe['productid'].nunique()))
    print("Number of users {}".format(dataframe['userid'].nunique()))
    print("Number of duplicate rows {}".format(len(dataframe[dataframe.duplicated(keep='first')])))

    print("\nREMOVING DUPLICATE ROWS")
    print(dataframe[dataframe.duplicated(keep=False)])
    dataframe = dataframe.drop_duplicates(keep='first')
    print("Total reviews {}".format(len(dataframe)))
    print("Number of products {}".format(dataframe['productid'].nunique()))
    print("Number of users {}".format(dataframe['userid'].nunique()))

    print('\nREMOVING REVIEWS WITH <a href="http", <span class="tiny"> Length, Warning, NOTE')
    dataframe = dataframe.drop(dataframe[dataframe['text'] == '<a href="http'].index)
    dataframe = dataframe.drop(dataframe[dataframe['text'] == '<span class="tiny"> Length'].index)
    dataframe = dataframe.drop(dataframe[dataframe['text'] == 'Warning'].index)
    dataframe = dataframe.drop(dataframe[dataframe['text'] == 'NOTE'].index)
    print("Total reviews {}".format(len(dataframe)))
    print("Number of products {}".format(dataframe['productid'].nunique()))
    print("Number of users {}".format(dataframe['userid'].nunique()))

    print("\nCLUSTERING PRODUCTS")
    print(dataframe[dataframe.duplicated(subset=['userid', 'text'], keep=False)])
    for text in dataframe[dataframe.duplicated(subset=['userid', 'text'], keep=False)]['text'].unique():
        product_ids = []
        for n, row in dataframe[dataframe['text'] == text].iterrows():
            product_ids.append(row['productid'])
        for duplicate in product_ids[1:len(product_ids)]:
            dataframe = dataframe.drop(dataframe[dataframe['productid'] == duplicate].index)
    print("Total reviews {}".format(len(dataframe)))
    print("Number of products {}".format(dataframe['productid'].nunique()))
    print("Number of users {}".format(dataframe['userid'].nunique()))
    dataframe.to_csv(os.path.join(WORKING_DIRECTORY, 'assets', "food.csv"))


def exploratory_analysis():
    dataframe = pd.read_csv(os.path.join(WORKING_DIRECTORY, "assets", "food.csv"), sep=',', index_col=0).reset_index(
        drop=True)
    print(dataframe)

    # Missingness Map
    fig = plt.figure(figsize=(7, 5))
    sns.heatmap(dataframe.isnull(), cbar=False)
    plt.xlabel("Column", fontdict={"size": 14})
    plt.ylabel("Row", fontdict={"size": 14})
    plt.title("Missingness Map", fontdict={"size": 20})
    plt.yticks(np.arange(0, len(dataframe), 5000), ['0', '5000', '10000', '15000', '20000', '25000', '30000'])
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(WORKING_DIRECTORY, 'assets', "heatmap.png"), format="png", dpi=1000)

    print('\nRATINGS')
    fives = list(dataframe['score']).count(5)
    fours = list(dataframe['score']).count(4)
    threes = list(dataframe['score']).count(3)
    twos = list(dataframe['score']).count(2)
    ones = list(dataframe['score']).count(1)
    print("5-STAR {}".format(fives))
    print("4-STAR {}".format(fours))
    print("3-STAR {}".format(threes))
    print("2-STAR {}".format(twos))
    print("1-STAR {}".format(ones))
    print(dataframe['score'].value_counts(normalize=True) * 100)

    # Distribution of Ratings
    print("RATINGS MEAN {}".format(np.mean(dataframe['score'])))
    print("RATINGS STD {}".format(np.std(dataframe['score'])))
    fig = plt.figure(figsize=(7, 4))
    plt.bar(['1', '2', '3', '4', '5'],
            dataframe['score'].value_counts(normalize=True).sort_index() * 100,
            color=['steelblue'])
    plt.xlabel("Score (stars)", fontdict={"size": 14})
    plt.ylabel("%", fontdict={"size": 14})
    plt.title("distribution of ratings".title(), fontdict={"size": 20})
    plt.grid(axis="x")
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(WORKING_DIRECTORY, 'assets', "ratings_distribution.eps"), format="eps")

    # Reviews per user
    print('\nREVIEWS PER USER')
    group_by_user = dataframe.groupby('userid')['userid'].count()
    print("REV. PER USER MEAN {}".format(np.mean(group_by_user)))
    print("REV. PER USER STD {}".format(np.std(group_by_user)))
    reviews_per_user = group_by_user[group_by_user < 10].value_counts().sort_index()
    print(reviews_per_user)
    fig = plt.figure(figsize=(7, 5))
    plt.bar(reviews_per_user.index, reviews_per_user,
            color=['steelblue'])
    plt.xlabel("Published reviews", fontdict={"size": 14})
    plt.ylabel("Count", fontdict={"size": 14})
    plt.title("Reviews per user".title(), fontdict={"size": 20})
    plt.grid(axis="x")
    plt.xticks(reviews_per_user.index)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(WORKING_DIRECTORY, 'assets', "reviews_per_user.eps"), format="eps")

    # # Reviews per product
    print('\nREVIEWS PER PRODUCT')
    group_by_product = dataframe.groupby('productid')['productid'].count()
    print("REV. PER PRODUCT MEAN {}".format(np.mean(group_by_product)))
    print("REV. PER PRODUCT STD {}".format(np.std(group_by_product)))
    reviews_per_product = group_by_product[group_by_product < 21].value_counts().sort_index()
    print(reviews_per_product)
    fig = plt.figure(figsize=(7, 5))
    plt.bar(reviews_per_product.index, reviews_per_product,
            color=['steelblue'])
    plt.xlabel("Received reviews", fontdict={"size": 14})
    plt.ylabel("Count", fontdict={"size": 14})
    plt.title("Reviews per product".title(), fontdict={"size": 20})
    plt.grid(axis="x")
    plt.xticks(reviews_per_product.index)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(WORKING_DIRECTORY, 'assets', "reviews_per_product.eps"), format="eps")

    # Features
    features_count = [4054, 3856, 3832, 2036, 1307, 1118, 978, 940, 921, 670, 584, 507, 357, 299, 290, 263, 245]

    features = [
        "flavor",
        "taste",
        "product",
        "price",
        "bag",
        "quality",
        "size",
        "box",
        "brand",
        "texture",
        "blend",
        "package",
        "variety",
        "color",
        "aroma",
        "smell",
        "ingredient"
    ]

    fig = plt.figure(figsize=(12, 5))
    plt.bar(features, features_count, color=['steelblue'])
    plt.xlabel("Feature", fontdict={"size": 14})
    plt.ylabel("Count", fontdict={"size": 14})
    plt.title("most frequent features".title(), fontdict={"size": 20})
    plt.grid(axis="x")
    plt.tick_params(axis='x', which='major', rotation=45)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(WORKING_DIRECTORY, 'assets', "most_frequent_features.eps"), format="eps")


def print_sentiment_results():
    dataframe = pd.read_csv(os.path.join(WORKING_DIRECTORY, 'assets', "sentiment-results-3.csv"), sep=',', index_col=0)
    dataframe['prediction'] = dataframe.apply(
        lambda x: 'positive' if x.prediction > 0 else 'negative' if x.prediction < 0 else 'neutral', axis=1)
    features = ["flavor", "taste", "product", "price", "quality", "brand", "texture", "package", "variety", "smell"]
    groups = []
    for feature in features:
        groups.append(
            dataframe[dataframe['feature'] == feature].prediction.value_counts().sort_index(ascending=False).tolist())
    group_labels = ["Flavor", "Taste", "Product", "Price", "Quality", "Brand", "Texture", "Package", "Variety", "Smell"]
    df = pd.DataFrame(groups, index=group_labels, columns=['Positive', 'Neutral', 'Negative'], )
    df.plot.bar(color=[(52 / 255, 199 / 255, 89 / 255),
                       (255 / 255, 149 / 255, 0 / 255),
                       (255 / 255, 59 / 255, 48 / 255)], figsize=(10, 4))
    plt.xticks(rotation=45)
    plt.grid(axis='x')
    plt.title('Sentiment Analysis Results with SO-CAL', fontdict={"size": 20})
    plt.tick_params(labelsize=14)
    plt.xlabel('Features', fontdict={"size": 14})
    plt.ylabel('Count', fontdict={"size": 14})
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(WORKING_DIRECTORY, 'assets', "sentiment-results.eps"), format="eps")


preprocessing()

exploratory_analysis()

es = ElasticSearchClient()
dataframe = es.get_statistics()
dataframe.to_csv(os.path.join(WORKING_DIRECTORY, 'assets', 'sentiment-results-3.csv'))
dataframe = pd.read_csv(os.path.join(WORKING_DIRECTORY, "assets", "sentiment-results-3.csv"), sep=',', index_col=0)
print(dataframe)

print_sentiment_results()
