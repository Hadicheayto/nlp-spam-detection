# NLP Project: Pre-Processing Textual Data, Detecting Spam Or Ham Messages
## Project made by: Hadi Cheayto

This project focuses on preprocessing textual data for spam detection in SMS messages. 
By cleaning and transforming raw text through techniques like tokenization, stopword removal, and lemmatization, we prepare the data for efficient classification.
These preprocessing steps are essential in NLP, as they help extract relevant features and improve the model's ability to accurately distinguish between spam and non-spam messages.


We selected this dataset to perform a range of preprocessing tasks, preparing the data for detecting spam versus ham messages.

Source data set: kaggle 
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Libraries used in our project:
#### libraries used during the pre-processing
1-pandas: A data manipulation library commonly used for handling tabular data and performing data preprocessing tasks.

2-numpy: A numerical computing library for handling arrays, mathematical functions, and large datasets.

3-re: The regular expressions module, used for text cleaning and pattern matching, crucial in text preprocessing.

4-warnings: Suppresses warnings that may arise during data processing or model training.

5-nltk: The Natural Language Toolkit, a comprehensive library for NLP tasks, including tokenization, stemming, and lemmatization.

6-stopwords (nltk.corpus): Part of NLTK, providing lists of common words to exclude in NLP tasks, improving model focus on meaningful terms.

7string: A standard Python library used for handling strings and removing punctuation.

8-word_tokenize (nltk.tokenize): Splits text into individual words or tokens, an essential step for NLP model preparation.

9-PorterStemmer (nltk.stem): A stemming algorithm in NLTK to reduce words to their root forms, aiding in consistency in text data.

10-WordNetLemmatizer (nltk.stem): A lemmatization tool in NLTK that converts words to their base forms, improving vocabulary standardization.
#### libraries used during the model development 

11- TfidfVectorizer (sklearn.feature_extraction.text): Converts text data into TF-IDF features, weighting terms by their importance in each document relative to the entire dataset, enhancing feature relevance.

12- train_test_split (sklearn.model_selection): Splits data into training and test sets, allowing for model training and evaluation on separate datasets.

13- SVC (sklearn.svm): Support Vector Classifier, a machine learning model for classification tasks, often effective in high-dimensional spaces like text data.

14- classification_report (sklearn.metrics): Generates a summary report of classification metrics, including precision, recall, and F1-score for model evaluation.
