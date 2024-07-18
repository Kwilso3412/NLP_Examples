# https://medium.com/towards-data-science/unsupervised-keyphrase-extraction-with-patternrank-28ec3ca737f0
# https://towardsdatascience.com/enhancing-keybert-keyword-extraction-results-with-keyphrasevectorizers-3796fa93f4db
# https://github.com/TimSchopf/KeyphraseVectorizers#installation

#pip install keyphrase-vectorizers
#pip install keybert

from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
import pandas as pd
import numpy as np

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
                keyword_phrases_list.append({f'{column_name}': keyword_list,
                                            "importance": keyword_importance})
                keyword_phrases = pd.DataFrame(keyword_phrases_list)
                keyword_phrases[f'{column_name}'] = keyword_phrases[f'{column_name}'].apply(', '.join)
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
                keywords_list.append({f'{column_name}': keyword_list,
                                            "importance": keyword_importance})
                keywords = pd.DataFrame(keywords_list)
                keywords[f'{column_name}'] = keywords[f'{column_name}'].apply(', '.join)
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
