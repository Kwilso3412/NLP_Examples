'''
## Keybert & keyphrase_vectorizers
## https://medium.com/towards-data-science/unsupervised-keyphrase-extraction-with-patternrank-28ec3ca737f0
## https://towardsdatascience.com/enhancing-keybert-keyword-extraction-results-with-keyphrasevectorizers-3796fa93f4db
## https://github.com/TimSchopf/KeyphraseVectorizers#installation

# pip install keyphrase-vectorizers
# pip install keybert

## NLTK 
## https://realpython.com/python-nltk-sentiment-analysis/

pip install nltk
nltk.download([
...     "names",
...     "stopwords",
...     "state_union",
...     "twitter_samples",
...     "movie_reviews",
...     "averaged_perceptron_tagger",
...     "vader_lexicon",
...     "punkt",
... ])


## Spacy
## Train your own
## https://mysteryweevil.medium.com/building-a-sentiment-analysis-model-using-spacy-a-practical-guide-261d881e5dcb
## https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397

# pip install spacy
# python -m spacy download en_core_web_sm
'''

from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
import pandas as pd
import numpy as np
import spacy 
from spacy.training.example import Example
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import random
from collections import Counter
import joblib

class nlp:
    def __init__(self):
        self.kw_model = KeyBERT()

    def keywords_ngram1(self, texts: list) -> list:
        '''
        Finds keywords of length 1 in the input texts.
            Args:
                texts (list): list of input texts
            Returns:
                list: one word keywords
            
            print('Keywords, ngram 1')
        '''
        try:
            kw = self.kw_model.extract_keywords(
                docs=texts, keyphrase_ngram_range=(1, 1))
        except ValueError as e:
            return None
        if len(texts) == 1:
            kw = [kw]
        keywords = []
        for kw_tuple in kw:
            try:
                keyword_list, keyword_importance = zip(*kw_tuple)
                keywords.append(list(keyword_list))
            except Exception as e:
                keywords.append(None)
        return keywords

    def keywords_keyphrase(self, texts: list) -> list:
        '''
        Generates keywords of multiple lengths from given text input
            Args:
                texts (list): list of text
            Returns:
                list: list of keyword phrases

            print(“Keywords, Phrases”)
        '''
        try:
            kw = self.kw_model.extract_keywords(
                docs=texts, vectorizer=KeyphraseCountVectorizer())
        except ValueError as e:
            return None
        if len(texts) == 1:
            kw = [kw]
        keyword_phrases = []
        for kw_tuple in kw:
            try:
                keyword_list, keyword_importance = zip(*kw_tuple)
                keyword_phrases.append(list(keyword_list))
            except Exception as e:
                keyword_phrases.append(None)
        return keyword_phrases
    
    def overall_keyphrases(insert_list):
        vectorizer = KeyphraseCountVectorizer()
        vectorizer.fit(insert_list)
        feature_names = vectorizer.get_feature_names_out()
        return feature_names


    '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    If you need it in a dataframe then you will use the following
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''

    # multi word
    def keyphrase(self, column_name , texts: list) -> list:
        '''
        Generates keywords of multiple lengths from given text input
            Args:
                texts (list): list of text
            Returns:
                list: list of keyword phrases

            print(“Keywords, Phrases”)
        '''
        try:
            kw = self.kw_model.extract_keywords(
                docs=texts, vectorizer=KeyphraseCountVectorizer())
        except ValueError as e:
            return None
        if len(texts) == 1:
            kw = [kw]
        keyword_phrases_list = []
        for kw_tuple in kw:
            try:
                keyword_list, keyword_importance = map(list, zip(*kw_tuple))
                keyword_phrases_list.append({'keyword': keyword_list,
                                            "importance": keyword_importance})
                keyword_phrases = pd.DataFrame(keyword_phrases_list)
                keyword_phrases['keyword'] = keyword_phrases['keyword'].apply(', '.join)
                keyword_phrases['total_importance'] = keyword_phrases['importance'].apply(sum).round(2)
                keyword_phrases['importance'] = keyword_phrases['importance'].apply(lambda x: '-'.join(map(str, x)))

            except Exception as e:
                keyword_phrases_list.append(None)
                keyword_phrases = pd.DataFrame(keyword_phrases_list)
                
        return keyword_phrases
    
    def keywords_ngram(self, column_name, texts: list) -> list:
        '''
        Finds keywords of length 1 in the input texts.
            Args:
                texts (list): list of input texts
            Returns:
                list: one word keywords
            
            print('Keywords, ngram 1')
        '''
        try:
            kw = self.kw_model.extract_keywords(
                docs=texts, keyphrase_ngram_range=(1, 1))
        except ValueError as e:
            return None
        if len(texts) == 1:
            kw = [kw]
        keywords_list = []
        for kw_tuple in kw:
            try:
                keyword_list, keyword_importance = map(list, zip(*kw_tuple))
                keywords_list.append({'keyword': keyword_list,
                                            "importance": keyword_importance})
                keywords = pd.DataFrame(keywords_list)
                keywords['keyword'] = keywords['keyword'].apply(', '.join)
                keywords['total_importance'] = keywords['importance'].apply(sum).round(2)
                keywords['importance'] = keywords['importance'].apply(lambda x: '-'.join(map(str, x)))
            except Exception as e:
                keywords.append(None)
        return keywords
    

    def overall_keywords_and_importance(insert_list):
        keyword_list = []
        try:
            vectorizer = KeyphraseCountVectorizer()
            list_matrix = vectorizer.fit_transform(insert_list)
            feature_names = vectorizer.get_feature_names_out()
            importance = list_matrix.sum(axis=0).A1
            importance = importance.tolist() if isinstance(importance, np.ndarray) else list(importance)

            for i, (keyword,rank) in enumerate(zip(feature_names,importance)):
                keyword_list.append({
                    'keyword': keyword,
                    'importance': rank
                })
            unsorted_keywords = pd.DataFrame(keyword_list)
            keywords = unsorted_keywords.sort_values('importance', ascending=False)
        
        except Exception as e:
                keyword_list.append(None)
                keywords = pd.DataFrame(keyword_list)
        return keywords 
    

    '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Train and deploy your own NLP
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''
    def report_of_model(self, nlp, data, stage):
        y_true = []
        y_pred = []
        for sample in data:
            doc = nlp(sample['text'])
            true_cats = sample['cats']
            pred_cats = doc.cats
            true_label = max(true_cats, key=true_cats.get)
            pred_label = max(pred_cats, key=pred_cats.get)
            y_true.append(true_label)
            y_pred.append(pred_label)

        print(f"\n Classification Report for {stage} Data:")
        print(classification_report(y_true, y_pred))


   def train_and_evaluate_model(nlp, train_data, num_epochs=10):
    '''
    Example.from_dict(doc, sample):
    This creates a spaCy Example object, which pairs the processed document (doc) with its correct labels (from sample). This is how spaCy knows what the correct output should be for this example.
    
    nlp.update([gold], drop=0.5): 
    This is where the actual learning happens. It updates the model's parameters based on this example. The drop=0.5 is a dropout rate, which helps prevent overfitting.
    '''
    # split the data
    x_train, x_test = train_test_split(train_data, test_size = 0.3, random_state=42)
    x_hold, x_val = train_test_split(x_test, test_size=0.5) 

    # randomize the data
    random.seed(42)
    # if there is not a predefined pipeline already added for example nlp.add_pipe("sentencizer") or nlp.add_pipe("parser")
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat")
        for label in ["positive", "negative", "neutral"]:
            textcat.add_label(label)
    else:
        textcat = nlp.get_pipe("textcat")

    optimizer = nlp.initialize() 

    # train the model 
    print('first round of training')
    for epoch in range(num_epochs):
        random.shuffle(train_data)
        for sample in x_train:
            doc = nlp.make_doc(sample['text'])
            gold = Example.from_dict(doc, sample)
            nlp.update([gold], sgd=optimizer, drop=0.5)
    print('\n Training Results: ')
    report_of_model(nlp, x_train, "Training Step")

    print('second round of training')
    for epoch in range(num_epochs):
        for sample in x_val:
            doc = nlp.make_doc(sample['text'])
            gold = Example.from_dict(doc, sample)
            nlp.update([gold], sgd=optimizer, drop=0.5)
    print('\n Validation Results')
    report_of_model(nlp, x_val, "validation Step")

    print('third round of training')
    for epoch in range(num_epochs):
        for sample in x_hold:
            doc = nlp.make_doc(sample['text'])
            gold = Example.from_dict(doc, sample)
            nlp.update([gold], sgd=optimizer, drop=0.5)
    print('\n Test Results')
    report_of_model(nlp,x_hold , "Final Test")

    joblib.dump(nlp, "sentiment_model.joblib")
    return nlp
    '''
    ## Sample usuage:
    nlp = spacy.blank("en")
    train_and_evaluate_model(nlp, train_data)
    '''
    ### string input, dictionary output
    def predict_sentiment_string(self, loaded_nlp, text):
        doc = loaded_nlp(text)
        scores = doc.cats
        predicted_class = max(scores, key=scores.get)
        return predicted_class, scores
    '''
    ## Sample usuage:
    text = "This new restaurant is amazing! The food is delicious and the service is excellent."

    predicted_class, scores = predict_sentiment(text)
    print(f'Predicted class: {predicted_class}')
    print(f'Scores: {scores}')
    '''

    ### list of dictionaries input, df output
    def predict_sentiment_listdict_to_df(loaded_nlp , data):
        results = []
        texts = [item['text']for item in data]
        doc = list(loaded_nlp.pipe(texts))

        for item, doc in zip(data, doc):
            scores = doc.cats
            positive, negative, neutral = scores.values()
            predicted_class = max(scores, key=scores.get)
            results.append({
                    'text': doc.text,
                    'predicted_class': predicted_class,
                    'positive':positive,
                    'negative':negative,
                    'neutral':neutral
            })

        results_df = pd.DataFrame(results)
        return results_df
    '''
    Sample Usuage: 

    loaded_nlp = joblib.load("sentiment_model.joblib")
    sentiment_results = predict_sentiment_listdict_to_df(loaded_nlp, new_train_data)
    sentiment_results
    '''

    # dataframe input, dataframe output
    def predict_sentiment_df(loaded_nlp , dataframe):
        results = []
        texts = dataframe['text']
        docs = list(loaded_nlp.pipe(texts))
        for docs in docs:
            scores = docs.cats
            positive, negative, neutral = scores.values()
            predicted_class = max(scores, key=scores.get)
            results.append({
                    'text': docs.text,
                    'predicted_class': predicted_class,
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral
            })
        results_df = pd.DataFrame(results)
        return results_df
    
    '''
    Sample Usuage: 
    loaded_nlp = joblib.load("sentiment_model.joblib")
    train_data_df = pd.DataFrame(new_train_data)
    sentiment_results = predict_sentiment_df(loaded_nlp, train_data_df)
    '''
