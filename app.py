import re
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from ElasticSearchClient import ElasticSearchClient
from math import floor, isnan

app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=["https://pro.fontawesome.com/releases/v5.10.0/css/all.css"])

navbar = dbc.NavbarSimple([
    dbc.NavItem(dbc.NavLink("Search Product", href='/search', className="text-white")),
    dbc.NavItem(dbc.NavLink("About", href='/about', className="text-white"))
],
    brand="Feature-Based Sentiment Analysis",
    brand_href="/search",
    color="primary",
    dark=True,
    style={"height": "65px"},
    className="sticky-top shadow bg-custom-2"
)

reviews_card = dbc.Card(
    dbc.CardBody(
        html.Div(
            html.H3('Please select one feature'),
            className=" h-100 d-flex align-items-center justify-content-center"
        ),
        className="h-100"
    ),
    className="h-100 shadow"
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

homepage = html.Div([
    dbc.Row([
        dbc.Col([
            navbar,
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(""),
                            width=3,
                            className="d-none d-lg-block"
                        ),
                        dbc.Col(
                            [
                                html.Div([
                                    dbc.Row([
                                        dbc.Col(
                                            'Feature-Based Sentiment Analysis',
                                            className="text-gradient text-center"
                                        )
                                    ], className="w-100", style={"line-height": "1"}),
                                    dbc.Row([
                                        dbc.Col(
                                            ' know this question is old. And the question did not mentioned which version of '
                                            'Bootstrap he was using. So ill assume the answer to this question is resolved.',
                                            className=" pr-3 pl-3 pt-3 pb-4 text-justify"
                                        )
                                    ], className="w-100"),
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Dropdown(placeholder="Select a product", id="search-dropdown"),
                                            width=10
                                        ),
                                        dbc.Col(
                                            dbc.Button("Search!", color="primary", className="mr-1", id="start-search"),
                                            width=2
                                        )
                                    ], className="w-100")
                                ], className="h-100 container-fluid",
                                    style={"padding-top": "35%"})
                            ],
                            className="w-100 col-lg-6 col-sm-12"
                        ),
                        dbc.Col(html.Div(""), width=3, className="d-none d-lg-block"),
                        html.Div(id="hidden-div")
                    ],
                    className="h-100"
                )
            ], className="container-fluid h-100")
        ], className="w-100 h-100 p-0")
    ], className="h-100")
],
    className="container-fluid vh-100 overflow-hidden"
)

product_page = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Store(id='memory', storage_type='local'),
            navbar,
            html.Div(id='my-div', style={"display": "none"}),
            dbc.Row([
                dbc.Col([],
                        id="first-card",
                        className=" col-4"),
                dbc.Col([],
                        id="second-card",
                        className="col-8")
            ],
                style={"height": "26%"},
                className="pt-3 pl-3 pr-3"
            ),
            dbc.Row([
                dbc.Col(
                    reviews_card,
                    id="third-card",
                    className="col-12")
            ],
                style={"height": "65%"},
                className="p-3"
            )
        ], className="p-0")
    ], className="h-100")
], className="container-fluid vh-100 overflow-hidden")

about_page = html.Div(
    [
        navbar
    ]
)


# Callback per ricercare i nomi dei prodotti nel db
@app.callback(
    [dash.dependencies.Output("search-dropdown", "options")],
    [dash.dependencies.Input("search-dropdown", "search_value")],
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    hits = es.get_products(search_value)
    return [hits]


# Callback per aprire la pagina del prodotto specifico
@app.callback(
    [Output("hidden-div", "children")],
    [Input('start-search', 'n_clicks')],
    [State('search-dropdown', 'value')]
)
def update_output_div(clicks, value):
    if clicks:
        return [dcc.Location(pathname="/product/" + value, id="someid_doesnt_matter")]
    else:
        raise PreventUpdate()


def make_product_card(product_id, product):
    stars = []
    for star in range(1, floor(product['rating']['avg_rating']) + 1):
        stars.append(html.I(className="fa fa-star fa-lg checked", style={"color": "orange"}))
    for star in range(len(stars) + 1, 6):
        stars.append(html.I(className="fa fa-star fa-lg "))
    keywords = []
    features_sentiment = []
    for feature in product['features']:
        if 'sentiment' in feature:
            features_sentiment.append(feature['sentiment'])
    if len(features_sentiment) > 0:
        product_sentiment = round(np.mean(features_sentiment), 2)
    else:
        product_sentiment = 0
    for keyword in product['keywords']:
        keywords.append(
            dbc.Row(html.H5(html.Span(keyword, className="badge badge-primary"), className="m-0"),
                    className="pl-3 pt-1")
        )
    if product_sentiment > 0:
        class_name_sentiment = "fas fa-caret-up fa-lg"
        style_sentiment = {"color": "green"}
    elif product_sentiment < 0:
        class_name_sentiment = "fas fa-caret-down fa-lg"
        style_sentiment = {"color": "red"}
    else:
        class_name_sentiment = "fas fa-caret-right fa-lg"
        style_sentiment = {"color": "orange"}
    sentiment_card = dbc.Card([
        dbc.CardHeader([
            html.H6('Overall Sentiment', className="m-0 text-center text-white card-title")
        ], className="bg-custom-2"),
        dbc.CardBody([
            dbc.Row([
                html.Span("{} ".format(abs(product_sentiment)) if not isnan(product_sentiment) else "0",
                          className=class_name_sentiment,
                          style=style_sentiment),
            ], className="d-flex justify-content-center align-items-center")
        ])
    ])
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H5(product_id, className="m-0 align-bottom card-title p-1"),
                                className="col-6 align-bottom"
                            ),
                            dbc.Col(
                                html.Div(stars, id="tooltip-target", className="float-right"),
                                className="col-6 text-right m-0 p-0"
                            ),
                            dbc.Tooltip(
                                [
                                    html.Div("Total Reviews: " + str(product['reviews']),
                                             className="text-left"),
                                    html.Div("Average Rating: " + str(round(product['rating']['avg_rating'], 2)),
                                             className="text-left"),
                                    html.Div("Five Stars: " + str(product['rating']['five_stars']),
                                             className="text-left"),
                                    html.Div("Four Stars: " + str(product['rating']['four_stars']),
                                             className="text-left"),
                                    html.Div("Three Stars: " + str(product['rating']['three_stars']),
                                             className="text-left"),
                                    html.Div("Two Stars: " + str(product['rating']['two_stars']),
                                             className="text-left"),
                                    html.Div("One Star: " + str(product['rating']['one_stars']),
                                             className="text-left")
                                ],
                                target="tooltip-target",
                                placement="right"
                            ),
                        ],
                        className="w-100 align-items-center")
                ],
                className="bg-custom-2 text-white"
            ),
            dbc.CardBody(
                dbc.Row([
                    dbc.Col(keywords, className="col-6"),
                    dbc.Col([
                        sentiment_card
                    ], className="col-6")
                ])
            )
        ],
        className="h-100 shadow"
    )


def make_feature_list(features):
    features_list = []
    for feature in features:
        if 'sentiment' in feature:
            if feature['sentiment'] > 0:
                emoji = html.Span(feature['sentiment'], className="fa fa-caret-up")
                className = "btn btn-success w-100 text-white btn"
            elif feature['sentiment'] < 0:
                emoji = html.Span(feature['sentiment'], className="fa fa-caret-down")
                className = "btn btn-danger w-100 text-white btn"
            else:
                emoji = html.Span(feature['sentiment'], className="fa fa-caret-right")
                className = "btn btn-warning w-100 text-white btn"
        else:
            emoji = ''
            className = "btn btn-info w-100 text-white btn"
        if 'sentiment' in feature:
            button = dbc.Col([
                html.Button(["{} ".format(feature['name'].capitalize()), emoji], className=className,
                            id='button-feature-' + feature['name'])
            ])
        else:
            button = dbc.Col([
                html.Button([feature['name'].capitalize() + ' ', emoji],
                            disabled=True,
                            className=className,
                            id='button-feature-' + feature['name'])
            ])
        features_list.append(button)
    return features_list


def make_feature_card(product):
    first_five_features = make_feature_list(product['features'][0:5])
    second_five_features = make_feature_list(product['features'][5:len(product['features'])])
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Features", className="card-title mb-0 p-1"),
                           className="bg-custom-2 text-white"),
            dbc.CardBody(
                [
                    dbc.Row(first_five_features, className="pt-0 pl-3 pr-3"),
                    dbc.Row(second_five_features, className="pt-3 pl-3 pr-3"),
                ],
                className="pb-0 pl-0 pr-0"
            )
        ],
        className="h-100 shadow"
    )


# Callback per mostrare la carta prodotto e la carta features
@app.callback(
    [Output('first-card', 'children'),
     Output('second-card', 'children'),
     Output('memory', 'product')],
    [dash.dependencies.Input('url', 'pathname')]
)
def display_page(pathname):
    product_id = pathname.split('/')[-1]
    product = es.get_product(product_id)
    product_card = None
    feature_card = None
    if product:
        product = product.to_dict()['hits']['hits'][0]['_source']
        product_card = make_product_card(product_id, product)
        feature_card = make_feature_card(product)
    return product_card, feature_card, product


def make_stars(rating):
    stars = []
    for star in range(1, int(rating) + 1):
        stars.append(html.I(className="fa fa-star checked", style={"color": "orange"}))
    for star in range(len(stars) + 1, 6):
        stars.append(html.I(className="fa fa-star checked"))
    return stars


def make_review(review, feature):
    if 'sentiment' in review and review['sentiment'] > 0:
        style = {"color": "green"}
        className = "fas fa-caret-up"
    elif 'sentiment' in review and review['sentiment'] < 0:
        style = {"color": "red"}
        className = "fas fa-caret-down"
    else:
        style = {"color": "orange"}
        className = "fas fa-caret-right"
    if 'text' in review:
        text = review['text']
        pat = r'\b\S*%s\S*\b' % re.escape(feature)
        matches = re.findall(pat, text)
        for match in matches:
            text = re.sub(r'(?<!\*){}(?!\*)'.format(match), '**{}**'.format(match), text)
    else:
        text = ""
    stars = make_stars(review['rating'])
    row = dbc.Row([
        dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span([html.B('Sentiment:  '),
                                       html.I(str(abs(review['sentiment'])) + ' ', className=className, style=style)
                                       ])
                        ],
                            className="float-left")
                    ],
                        className="col-6 text-right"),
                    dbc.Col([
                        html.Div(stars,
                                 className="float-right")
                    ],
                        className="col-6 text-right m-0 p-0"),
                ],
                    className="w-100 align-items-end")
            ]),
            dbc.CardBody([
                dcc.Markdown(
                    text, className="text-justify m-0")
            ])
        ], className="w-100 mb-3 mr-3 ml-3")
    ])
    return row


def make_reviews_card(product, feature_name):
    features = product['features']
    flavor = None
    for feature in features:
        if feature['name'] == feature_name:
            flavor = feature
    positive_reviews = []
    if 'positive_sentences' in flavor:
        flavor['positive_sentences'].sort(key=lambda x: x['sentiment'], reverse=True)
        for review in flavor['positive_sentences']:
            positive_reviews.append(make_review(review, feature_name))
    neutral_reviews = []
    if 'neutral_sentences' in flavor:
        for review in flavor['neutral_sentences']:
            neutral_reviews.append(make_review(review, feature_name))
    negative_reviews = []
    if 'negative_sentences' in flavor:
        flavor['negative_sentences'].sort(key=lambda x: x['sentiment'])
        for review in flavor['negative_sentences']:
            negative_reviews.append(make_review(review, feature_name))
    positive_indicator = None
    negative_indicator = None
    if len(positive_reviews) > 0:
        positive_indicator = html.Div([
            html.I(flavor['positive_sentiment'],
                   className="fas fa-caret-up fa-lg",
                   style={"color": "white"})],
            className="float-right")
    if len(negative_reviews) > 0:
        negative_indicator = html.Div([
            html.I(abs(flavor['negative_sentiment']),
                   className="fas fa-caret-down fa-lg",
                   style={"color": "white"})],
            className="float-right")
    return dbc.Card([
        dbc.CardHeader([
            html.H5(html.B("Sentiment about {}: {}".format(feature_name.capitalize(), str(flavor['sentiment']))),
                    className="card-title mb-0 p-1"),
        ], className="bg-custom-2 text-white"),
        dbc.CardBody(
            [
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Positive Reviews ({})".format(str(len(positive_reviews))),
                                                className="align-bottom m-0"),
                                    ], className="col-8"),
                                    dbc.Col([
                                        positive_indicator
                                    ], className="col-4")
                                ], className="aling-items-end")
                            ], className="bg-success text-white"),
                            dbc.CardBody([
                                html.Div([positive_reviews[i] for i, review in enumerate(positive_reviews)])
                            ], className="overflow-auto", style={"height": "360px"})
                        ], className=" shadow")
                    ],
                        className="col-4"),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Neutral Reviews ({})".format(str(len(neutral_reviews))),
                                                className="align-bottom m-0"),
                                    ], className="col-8")
                                ], className="aling-items-end")
                            ], className="bg-warning text-white"),
                            dbc.CardBody([
                                html.Div([neutral_reviews[i] for i, review in enumerate(neutral_reviews)])
                            ], className="overflow-auto", style={"height": "360px"})
                        ], className=" shadow")
                    ],
                        className="col-4"),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Negative Reviews ({})".format(str(len(negative_reviews))),
                                                className="align-bottom m-0"),
                                    ], className="col-8"),
                                    dbc.Col([
                                        negative_indicator
                                    ], className="col-4")
                                ], className="aling-items-end")
                            ], className="bg-danger text-white"),
                            dbc.CardBody([
                                html.Div([negative_reviews[i] for i, review in enumerate(negative_reviews)])
                            ], className="overflow-auto", style={"height": "360px"})
                        ], className=" shadow")
                    ],
                        className="col-4")
                ], className="p-0")
            ]
        )],
        className="h-100 shadow"
    )


@app.callback(
    Output('third-card', 'children'),
    [Input('button-feature-flavor', 'n_clicks'),
     Input('button-feature-taste', 'n_clicks'),
     Input('button-feature-brand', 'n_clicks'),
     Input('button-feature-quality', 'n_clicks'),
     Input('button-feature-price', 'n_clicks'),
     Input('button-feature-product', 'n_clicks'),
     Input('button-feature-variety', 'n_clicks'),
     Input('button-feature-package', 'n_clicks'),
     Input('button-feature-smell', 'n_clicks'),
     Input('button-feature-texture', 'n_clicks')],
    [State('memory', 'product')]
)
def update_output(_b1, _b2, _b3, _b4, _b5, _b6, _b7, _b8, _b9, _b10, product):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered if p['value'] is not None]
    if len(changed_id) > 0:
        changed_id = changed_id[0]
    if 'button-feature-flavor' in changed_id:
        return make_reviews_card(product, 'flavor')
    elif 'button-feature-taste' in changed_id:
        return make_reviews_card(product, 'taste')
    elif 'button-feature-brand' in changed_id:
        return make_reviews_card(product, 'brand')
    elif 'button-feature-quality' in changed_id:
        return make_reviews_card(product, 'quality')
    elif 'button-feature-price' in changed_id:
        return make_reviews_card(product, 'price')
    elif 'button-feature-product' in changed_id:
        return make_reviews_card(product, 'product')
    elif 'button-feature-package' in changed_id:
        return make_reviews_card(product, 'package')
    elif 'button-feature-variety' in changed_id:
        return make_reviews_card(product, 'variety')
    elif 'button-feature-smell' in changed_id:
        return make_reviews_card(product, 'smell')
    elif 'button-feature-texture' in changed_id:
        return make_reviews_card(product, 'texture')
    else:
        raise PreventUpdate


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if '/product' in pathname:
        return product_page
    elif pathname == '/about':
        return about_page
    elif pathname == '/search':
        return homepage
    else:
        return homepage


if __name__ == '__main__':
    es = ElasticSearchClient()
    app.run_server(debug=True)
