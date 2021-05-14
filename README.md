# Feature-based Sentiment Analysis on Food Reviews
[![made-with-python](https://img.shields.io/badge/MADE%20WITH-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![made-with-dash](https://img.shields.io/badge/MADE%20WITH-Dash%20-%23EE4D2A.svg?&style=for-the-badge&logo=Plotly&logoColor=white)](https://plotly.com/dash/)
[![made-with-elasticsearch](https://img.shields.io/badge/MADE%20WITH-Elasticsearch%20-%2312D833.svg?&style=for-the-badge&logo=Elasticsearch&logoColor=white)](https://www.elastic.co/elasticsearch/)


<div align="justify">
This project aims to develop a web application that can help e-commerce users and product suppliers understand which are the strengths and weaknesses of products sold online. This information is the result of applying a feature-based sentiment analysis on customer reviews coming from Amazon. The used approach is based on three main steps: features extraction from products, opinion orientation identification, and results' summarization. Regarding opinion orientation identification, two methods were compared. The first is based on "The Semantic Orientation CALculator" (SO-CAL) framework while the other relies on the "for Valence Aware Dictionary for sEntiment Reasoning" (VADER) framework.
</div>

## Requirements
To install the requirements:

    pip install -r requirements.txt

## Data
The project is based on a reduced version of the Amazon Fine Food Reviews dataset (https://www.kaggle.com/snap/amazon-fine-food-reviews), which originally included about 500000 food reviews coming from a period of over ten years (until October 2012).
The smaller version is made up of a 35172 reviews and each of them contains the product's id, the user's id, the rating score given by him and, finally, the review's text.


## Usage
* <i>script.py</i> includes all the functions used for dataset preprocessing and figure generation.
* <i>main.py</i> includes all the function used to perform sentiment analysis and storing the results in Elasticsearch.
* <i>app.py</i> contains the web-app code.
* <i>preprocess.py</i>, <i>SO_Calc.py</i> and <i>SO_Run.py</i> and the directory <i>Resources</i> are adapted from the SO-CAL python library (https://github.com/sfu-discourse-lab/SO-CAL)



## References


* J.  J.  McAuley  and  J.  Leskovec,  “From  amateurs  to  connoisseurs:   modeling  theevolution of user expertise through online reviews,” inProceedings of the 22nd in-ternational conference on World Wide Web, 2013, pp. 897–908.
* B. Liuet al., “Sentiment analysis and subjectivity.”Handbook of natural languageprocessing, vol. 2, no. 2010, pp. 627–666, 2010.
* M. Hu and B. Liu, “Mining and summarizing customer reviews,” inProceedings ofthe tenth ACM SIGKDD international conference on Knowledge discovery and datamining, 2004, pp. 168–177.
* M. Eirinaki, S. Pisal, and J. Singh, “Feature-based opinion mining and ranking,”Journal of Computer and System Sciences, vol. 78, no. 4, pp. 1175–1184, 2012. 
* M. Taboada, J. Brooke, M. Tofiloski, K. Voll, and M. Stede, “Lexicon-based methodsfor sentiment analysis,”Computational linguistics, vol. 37, no. 2, pp. 267–307, 2011.
* C. Hutto and E. Gilbert, “Vader:  A parsimonious rule-based model for sentimentanalysis of social media text,” inProceedings of the International AAAI Conferenceon Web and Social Media, vol. 8, no. 1, 2014.
* R.  Campos,  V.  Mangaravite,  A.  Pasquali,  A.  Jorge,  C.  Nunes,  and  A.  Jatowt,“Yake!   keyword  extraction  from  single  documents  using  multiple  local  features,”Information Sciences, vol. 509, pp. 257–289, 2020.

## Authors
* Lorenzo Pirola &nbsp;
[![gmail](https://img.shields.io/badge/Gmail-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:l.pirola13@campus.unimib.it) &nbsp;
[![github](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/lpirola13) &nbsp;
[![linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lorenzo-pirola-230275197/)
