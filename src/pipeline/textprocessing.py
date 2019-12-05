# -*- coding: utf-8 -*-
# Text processing

'''
Functions needed to do text processing
'''
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
pipeline_path = str(Path(os.path.abspath(__file__)).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

# +
# Packages
from utils.utils import connect_to_database
from nltk.corpus import stopwords
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import re
import numpy as np
import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import os
import sys
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords',quiet=True)

from pipeline.data_persistence import persist_local, get_local, generate_id, check_if_local_exists

document_path = '/data/shared_data/data/raw/txt_extracted/'

def get_file_dict(id_llamado):
    """
    Obtain a dictionary of full filepaths given the input IDs

    Parameters
    ----------
    id_llamado : list
        list of IDs to obtain the full filepaths

    Returns
    -------
    file_path_dict : dictionary
        A dictionary of IDs as keys and their corresponding full path 
    """
    con = connect_to_database()

    query = """
    select id_llamado, filename, method
    from semantic.documents
    where is_extractable = true and method = 'tika'
    """

    # Load full list of all id_llamados and their corresponding filenames
    full_list = pd.read_sql_query(query, con)
    
    #Get the subset based on id_llamados
    subset_list = full_list[full_list['id_llamado'].isin(id_llamado)]

    #Change the filename to .txt instead of .pdf or .PDF
    if len(subset_list) != 0:
        subset_list['filename'] = subset_list.apply(lambda x: x['filename'].lower().replace('.pdf','')+'.txt', axis=1)
    else:
        print('Empty dataframe')

    # Generate the full path
    subset_list['fullpath'] = document_path + \
        subset_list['method'] + '/' + subset_list['filename']

    # Extract full path and id_llamado
    file_path_dict = {}
    file_path_dict['id_llamado'] = subset_list['id_llamado'].tolist()
    file_path_dict['fullpath'] = subset_list['fullpath'].tolist()

    return file_path_dict


def clean_string(text):
    """
    Performs basic string cleaning:
    - Clear pdf conversion md strings
    - Remove weird symbols
    - Reduce redundant white spaces

    Parameters
    ----------
    text : list
        list of strings to clean

    Returns
    -------
    text : list
        list of strings that are cleaned by the above process
    """
    # Clear line breaks
    md_break = re.compile('(\\n|\\uf0d8|\\uf0b7|\\uf0fc|\\uf020)')
    text = re.sub(md_break, ' ', text)

    # Clear symbols and numbers
    text = re.sub(
        re.compile(
            '[0-9–°“”�º…' + string.punctuation + ']'),
        ' ',
        text)

    # Clear multiple whitespace
    text = re.sub(re.compile(r'\s+'), ' ', text)
    return text


def split_dictionary(output_dict):
    """Splits a dictionary into two lists for IDs and its full file paths
    
    Parameters
    ----------
    output_dict : dictionary
        A dictionary with keys: 'id_llamado' and 'fullpath'
    
    Returns
    -------
    Two lists
        Two lists of 'id_llamado' and 'fullpath'
    """
    id_llamado = output_dict['id_llamado']
    filenames = output_dict['fullpath']
    return filenames, id_llamado


def cleaned_text_extracts(filenames):
    """Open and read the files and clean the texts and save them as a list of strings.
    
    
    Parameters
    ----------
    filenames : list
        A list of full file paths
    
    Returns
    -------
    list
        A list of cleaned strings each corresponding to a .txt file
    """
    all_docs = []
    tracker = 0
    for txt_file in filenames:
        with open(txt_file) as f:
            txt_file_as_string = clean_string(f.read())
            # print(txt_file_as_string[:3])
        all_docs.append(txt_file_as_string)
        tracker += 1
        # print(f'opened: {tracker}')
    return all_docs


def vector_fit(text_list, args, stop_words):
    """Fits a TF-IDF algorithm to a list of strings corresponding to various .txt files
    
    Parameters
    ----------
    text_list : list
        List of strings corresponding to each .txt files
    args : dictionary
        Dictionary containing the parameters to the TF-IDF algorithm
    stop_words : list
        A list of stop_words from the NLTK package
    
    Returns
    -------
    TFIDFVectorizer object
        An object containing the trained parameters of the TFIDFVectorizer
    """
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    vectorizer.set_params(**args)
    vec_fit = vectorizer.fit(text_list)
    return vec_fit


def vector_transform(text_list, id_llamado, vec_fit):
    """Transforms the list of texts into features using a trained TFIDFVectorizer object
    
    Parameters
    ----------
    text_list : list
        list of strings corresponding to each .txt file
    id_llamado : list
        list of IDs corresponding to the .txt file
    vec_fit : TFIDFVectorizer object
        trained encoder using training data
    
    Returns
    -------
    Dataframe
        A dataframe containing the ID and the TFIDF features. One column represent one word.
    """
    vector_doc = vec_fit.transform(text_list)
    vec = vector_doc.toarray()
    # place tf-idf values in a pandas data frame
    df = pd.DataFrame(
        vec,
        index=id_llamado,
        columns=vec_fit.get_feature_names())
    df = df.rename_axis('id_llamado').reset_index()
    return df

# Workflow functions

def tfidf_preprocess(id_llamado):
    """Wrapper function that preprocess texts in preparation for TFIDF 
    
    Parameters
    ----------
    id_llamado : list of IDs
        list of id_llamado to be processed
    
    Returns
    -------
    two lists
        id_llamado is the list of ids cleaned
        text_files_list is the list of strings corresponding to each ID after cleaning the .txt file
    """
    # Get list of id_llamado in question and their corresponding full path
    filelist_dict = get_file_dict(id_llamado)
    filenames, id_llamado = split_dictionary(filelist_dict)

    # Extract the texts using filename paths into list
    text_files_list = cleaned_text_extracts(filenames)

    return id_llamado, text_files_list


def run_tfidf(fold, args):
    """Wrapper function that runs the process of TFIDF after preprocessing
    
    Parameters
    ----------
    fold : dict
        id_llamado lists to filter features and labels
    args : dict
        dictionary of parameters to be passed into the TFIDF algo
    
    Returns
    -------
    pd.DataFrame
        train and test dataframes for train and test document features
    """

    fold_id = {'fold_id': generate_id(str(fold) + str(args['params']))}

    if check_if_local_exists(fold_id, 'tfidf-train', ['fold_id']):
        tfidf_features_train = get_local(fold_id, 'tfidf-train', id_keys=['fold_id'], as_type='.parquet.gz')
        tfidf_features_test =  get_local(fold_id, 'tfidf-test', id_keys=['fold_id'], as_type='.parquet.gz')
   
    else:
        # Get the processed list of texts for both train and test
        train_id, train_text = tfidf_preprocess(fold['train'])
        test_id, test_text = tfidf_preprocess(fold['test'])
        
        # Get train and test document features sets
        stop_words = set(stopwords.words('spanish'))
        # Get TFIDF encoder
        tfidf_encode = vector_fit(train_text,args['params'],stop_words)
        # Get train and test dataframes
        tfidf_features_train = vector_transform(train_text,train_id,tfidf_encode)
        tfidf_features_test = vector_transform(test_text,test_id,tfidf_encode)

        
        persist_local(tfidf_encode, args, 'tfidf', ['experiment_id'], as_type='.p')

        persist_local(tfidf_features_train, fold_id, 'tfidf-train', ['fold_id'])
        persist_local(tfidf_features_test, fold_id, 'tfidf-test', ['fold_id'])

    return tfidf_features_train, tfidf_features_test
