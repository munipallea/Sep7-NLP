# Sentiment Analysis, NLP and Named Entity Recognition

__Title__: Tales from crypto <br />
__Submitted by__: Amar Munipalle <br />

## 1. Introduction and Technical Tools

The attached Jupyter notebook uses several Natural Language Processing dependencies to access sentiment evolution around two popular crypto currencies - Bitcoin and Ethereum.
Imports include (descriptions sourced from official documentation):
    NLTK: NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.

    Vader sentiment analysis: VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. It is fully open-sourced under the [MIT License] (we sincerely appreciate all attributions and readily accept most contributions, but please don’t hold us liable).

    Spacy for Named Entity Recognition: Spacy is an Industrial-Strength Natural Language Processing import in Python

    Newsapi:  per site is the unofficial Python client library to integrate News API into your Python application without worrying about what's going on under the hood.

* [Newsapi](https://newsapi.org/)
* [spacy](https://spacy.io/)
* [NLTK](https://www.nltk.org/)


## 2. Sentiment Analysis

![Sentimental](images/sentimental.jpeg)

The Newsapi client is used to download articles with Bitcoin and Ethereum keywords between certain dates (Aug 2020). The resulting object is manipulated and cleaned using a function call to store content of article in a Pandas dateframe to facilitate cleaning, analyis and analytics including sentiment analysis.
To keep code dry this implementation is through a function call. Extensive use of lambda functions, regex and .apply methods to manipulate and clean data.
In summary:
Bitcoin has highest mean positive score (0.079947) vs. Ethereum (0.051500)
Ethereum has the highest compound score (mean and max .16/-.08 .81/.79)
Ethereum has highest positive score (max .27/.21)
#### Neutral Sentiment dominates. Significantly higher mean, max and quartlies.

<p align="center">
<img src="images/eth_sent.png" width="600" height="300"/>
</p>

<p align="center">
Figure 1.Ethereum Sentiment Distribution
</p>

<p align="center">
<img src="images/bit_sent.png" width="600" height="300"/>
</p>

<p align="center">
Figure 2.Bitcoin Sentiment Distribution
</p>

<p align="center">
<img src="images/bit_sent_time.png" width="600" height="300"/>
</p>

<p align="center">
Figure 3. Bitcoin Sentiment Over Time
</p>


## 3. Natural Language Processing N-grams and Frequency Analysis
Using common NLTK algorithms like tokenization (decomposing into words and sentences), lemmatization (root words), stop words (noise and filler words in texts) and named entity recognition important pairwise distributions of words (N-grams) are generated.
Certain contextual words like (Reuters, chars etc.) are added to stop word corpus given their occurence and lack of informational value in articles.


## 4. Wordclouds

Wordclouds present a visual graphic of word tokens. By generating in a multi colour format key themes and messages can be extracted and conveyed to an executive audience.
Ethereum WC is enclosed below. Second order inferences (apart from obvious dominance of words like ) include connectedness (Network), recent market volatility (valuation, trading) and potential applications like DeFi and decentralized finance.
<p align="center">
<img src="images/Eth_WC.png" width="600" height="300"/>
</p>

<p align="center">
Figure 4.Ethereum WC
</p>

#### It is interesting to note the 'dont trust' theme in Bitcoin relates to its negative sentiment (relatively)

<p align="center">
<img src="images/bit_wc.png" width="600" height="300"/>
</p>

<p align="center">
Figure 2.Bitcoin Wordcloud
</p>


## 4. Named Entity Recognition

Per spacy, a named entity is a “real-world object” that’s assigned a name – for example, a person, a country, a product or a book title. spaCy can recognize various types of named entities in a document, by asking the model for a prediction. Because models are statistical and strongly depend on the examples they were trained on, this doesn’t always work perfectly and might need some tuning later, depending on your use case.

* [Named Entity Recognition](https://spacy.io/usage/linguistic-features#named-entities)

Prior to commencing NER a significant amount of cleanup is completed on the raw text data. This is similar to the sentiment analysis and includes lemmatization, tokenization and stop words. Jupyter notebook includes rich output on the two coins and associated NER

A quick analysis of the bitcoin NER posits as to why sentiment is negative - several articles covering fraud (NK hackers) and scams likely contributed to a low NPR
