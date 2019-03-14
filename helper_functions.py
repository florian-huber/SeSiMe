"""
Small helperfunctions for the omics project (NLeSC, 2018, 2019..)

@author: FlorianHuber
"""

from __future__ import print_function
import numpy as np
from scipy import spatial
import json
import math
import pandas as pd

##
## ---------------- Document processing functions -----------------------------
## 

def preprocess_document(corpus, stopwords, min_frequency = 2):
    """ Basic preprocessing of document words
    
    - Remove common words from stopwords and tokenize
    - Only include words that appear at least *min_frequency* times. Default = 2
    - Set words to lower case.
    """

    texts = [[word.lower() for word in document if word not in stopwords] for document in corpus]

    # remove words that appear only once
    from collections import defaultdict
    
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    texts = [[token for token in text if frequency[token] >= min_frequency] for text in texts]
    
    return texts, frequency




def create_distance_network(Cdistances_ids, Cdistances, filename="word2vec_test.graphml", 
                            cutoff_dist = 0.1,
                            max_connections = 25,
                            min_connections = 2):
    """ Built network from closest connections found
        Using networkx
        
        TODO: Add maximum number of connections 
    """
    
    dimension = Cdistances_ids.shape[0]
    
    # Form network
    import networkx as nx
    Bnet = nx.Graph()               
    Bnet.add_nodes_from(np.arange(0,dimension))   
    
    for i in range(0,dimension):      
#        idx = Cdistances_ids[i, (Cdistances[i,:] < cutoff_dist)]
        idx = np.where(Cdistances[i,:] < cutoff_dist)[0]
        if idx.shape[0] > max_connections:
            idx = idx[:(max_connections+1)]
        if idx.shape[0] <= min_connections:
            idx = np.arange(0, (min_connections+1))
        new_edges = [(i, int(Cdistances_ids[i,x]), float(Cdistances[i,x])) for x in idx if Cdistances_ids[i,x] != i]
        Bnet.add_weighted_edges_from(new_edges)
#        Bnet.add_edge(i, int(candidate), weight=float((max_distance - distances[i,candidate])/max_distance) )
        
    # export graph for drawing (e.g. using Cytoscape)
    nx.write_graphml(Bnet, filename)
    return Bnet



##
## ---------------- General functions ----------------------------------------
## 

def dict_to_json(mydict, file_json): 
    # save dictionary as json file
    with open(file_json, 'w') as outfile:  
        json.dump(mydict, outfile)

        
def json_to_dict(file_json): 
    # create dictionary from json file
    with open(file_json) as infile:  
        mydict = json.load(infile)

    return mydict


def full_wv(vocab_size, word_idx, word_count):
    """ Create full word vector
    """
    one_hot = np.zeros((vocab_size)) 
    one_hot[word_idx] = word_count
    return one_hot



##
## ---------------- Clustering & metrics functions ----------------------------
## 

def ifd_scores(vocabulary, corpus):
    """ Calulate idf score (Inverse Document Frequency score) for all words in vocabulary over a given corpus
    
    
    
    Args:
    --------
    vocabulary: gensim.corpora.dictionary
        Dictionary of all corpus words
    corpus: list of lists
        List of all documents (document = list of words)
    
    Output: 
        idf_scores: pandas DataFrame 
            contains all words and their ids, their word-count, and idf score
    """ 
    #TODO: this function is still slow! (but only needs to be calculated once)
    
    idf_scores = []
    idf_score = []
    vocabulary_size = len(vocabulary)
    corpus_size = len(corpus)
    
    for i in range(0, vocabulary_size):
        if (i+1) % 100 == 0 or i == vocabulary_size-1:  # show progress
                print('\r', ' Calculated scores for ', i+1, ' of ', vocabulary_size, ' words.', end="")
        
        word_containing = 0
        word = vocabulary[i]
        for document in corpus:
            word_containing += 1 * (document.count(word) > 0)
            idf_score = math.log(corpus_size / (max(1, word_containing)))   
            
        idf_scores.append([i, word, word_containing, idf_score])
    print("")
    return pd.DataFrame(idf_scores, columns=["id", "word", "word count", "idf score"])




def calculate_similarities(vectors, num_hits=25, method='cosine'):
    """ Calculate similarities (all-versus-all --> matrix) based on array of all vectors
    
    Args:
    -------
    num_centroid_hits: int
        Function will store the num_centroid_hits closest matches. Default is 25.      
    method: str
        See scipy spatial.distance.cdist for options. Default is 'cosine'.
        
    TODO: Check how to go from distance to similarity for methods other than cosine!!
    """
    Cdist = spatial.distance.cdist(vectors, vectors, method)
    mean_similarity = 1 - np.mean(Cdist)
    # Create numpy arrays to store distances
    list_similars_ids = np.zeros((Cdist.shape[0],num_hits), dtype=int)
    list_similars = np.zeros((Cdist.shape[0],num_hits))
    
    for i in range(Cdist.shape[0]):
        list_similars_ids[i,:] = Cdist[i,:].argsort()[:num_hits]
        list_similars[i,:] = 1- Cdist[i, list_similars_ids[i,:]]
    
    return list_similars_ids, list_similars, mean_similarity


