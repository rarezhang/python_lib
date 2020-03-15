# Python libs I used most  

## search engine  
- pylucene: https://lucene.apache.org/pylucene/ (low-level indexing and searching capability)  
    + [examples](https://github.com/rarezhang/python_lib/tree/master/examples/pylucene_test_python3)
- solr: https://lucene.apache.org/solr/guide/8_4/installing-solr.html (search server, built on top of lucene)  
    + tutorial: https://lucene.apache.org/solr/guide/8_4/solr-tutorial.html#solr-tutorial  
    + python: https://lucene.apache.org/solr/guide/8_4/using-python.html   


## web scraping  
- scrapy: https://docs.scrapy.org/en/latest/ (downloading, cleaning and saving data from the web)  
- beautifulsoup4: https://www.crummy.com/software/BeautifulSoup/bs4/doc/  (get information out of webpages)  
- requests: https://github.com/psf/requests/ (get the webpage)  
- tweepy: http://docs.tweepy.org/en/latest/index.html  

## NLP  
- NLTK: https://www.nltk.org/ (string processing: takes strings as input and returns strings or lists of strings as output)  
- spacy: https://spacy.io/ (object-oriented approach: returns document object whose words and sentences are objects themselves. Performance is better)  
  + spacy.displacy: https://spacy.io/usage/visualizers
- gensim: https://radimrehurek.com/gensim/ (word2vec, doc2vec, LDA, distance metrics, text summarization, pivoted document length normalization)  
- flashtext: https://github.com/vi3k6i5/flashtext (fast, replace keywords in sentences or extract keywords from sentences)  
- bert-pytorch: https://github.com/codertimo/BERT-pytorch (Train your own BERT model)  

## machine learning & data science 
- pandas: https://pandas.pydata.org/   
- scipy: https://www.scipy.org/  
- numpy: https://numpy.org/  
- scikit-learn: https://scikit-learn.org/stable/  
    + categorical-encoding: https://github.com/scikit-learn-contrib/categorical-encoding (different encoding methods)  
    + imbalanced-learn: https://github.com/scikit-learn-contrib/imbalanced-learn#imbalanced-learn (under-sampling, over-sampling, combining over-/under-sampling, create ensemble balanced sets)  
- statsmodels: https://www.statsmodels.org/stable/index.html (cross-sectional models, time-series models)  
- prefixspan: https://github.com/chuanconggao/PrefixSpan-py (PrefixSpan: frequent sequential pattern mining, BIDE: frequent closed sequential pattern mining (in closed.py), FEAT: frequent generator sequential pattern mining algorithm  (in generator.py))  
- imbalanced-learn: https://github.com/scikit-learn-contrib/imbalanced-learn (re-sampling techniques: under-sampling, over-sampling, combining over-/under-sampling, create ensemble balanced sets)     


## time series  
- PyFlux: https://pyflux.readthedocs.io/en/latest/index.html
    + **only python 3.5 so far**

## math  
- sympy: https://www.sympy.org/en/index.html (symbolic mathematics, computer algebra system)  

## deep learning  
- pytorch: https://pytorch.org/get-started/locally/  

## data base
- pymongo: https://api.mongodb.com/python/current/index.html  
- pymysql: https://github.com/PyMySQL/PyMySQL  


## system 
- cython: https://cython.readthedocs.io/en/latest/index.html  
- Jupyter: https://ipython.org/  


## network analysis 
- networkx: https://networkx.github.io/  
  + networkx algorithms: https://networkx.github.io/documentation/stable/reference/algorithms/index.html  
- graph-tool: https://graph-tool.skewed.de/ (manipulation and statistical analysis of network: fast, based on C++)  
  + layout: https://graph-tool.skewed.de/static/doc/draw.html  
  + centrality: https://graph-tool.skewed.de/static/doc/centrality.html  
  + **fully native installation on windows is not supported**, installing using Docker: https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#installing-using-docker  
  
## image  
- opencv4: https://docs.opencv.org/master/d6/d00/tutorial_py_root.html  
  + install: https://anaconda.org/conda-forge/opencv  
  + https://pypi.org/project/opencv-python/  
  + import cv2  
  
## visualizations  
- matplotlib: https://matplotlib.org/  
- seaborn: https://seaborn.pydata.org/ (based on matplotlib, provides a high-level interface)  
- bokeh: https://docs.bokeh.org/en/latest/index.html (interactive visualization)  
- plotly: https://plot.ly/python/ (basic charts, scientific charts, financial charts, maps, 3D charts, subplots, Jupyter Widgets Interaction, transforms, custom controls, animations)  






Bert  

lime  
shap  
py-spy  
