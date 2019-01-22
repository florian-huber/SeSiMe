"""
Small helperfunctions for the omics project (NLeSC, 2018, 2019..)

@author: FlorianHuber
"""

from __future__ import print_function
import numpy as np
from scipy import spatial

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





import json
import math
import pandas as pd

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



def get_spaced_colors(n):
    """ Create set of 'n' well-distinguishable colors
    """
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    RGB_colors = [(int(i[:2], 16)/255, int(i[2:4], 16)/255, int(i[4:], 16)/255) for i in colors]    

    return RGB_colors



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




def calculate_distances(vectors, num_hits=25, method='cosine'):
    """ Calculate distances (all-versus-all --> matrix) based on array of all vectors
    
    Args:
    -------
    num_centroid_hits: int
        Function will store the num_centroid_hits closest matches. Default is 25.      
    method: str
        See scipy spatial.distance.cdist for options. Default is 'cosine'.
    
    """
    Cdist = spatial.distance.cdist(vectors, vectors, method)
    
    # Create numpy arrays to store distances
    Cdistances_ids = np.zeros((Cdist.shape[0],num_hits), dtype=int)
    Cdistances = np.zeros((Cdist.shape[0],num_hits))
    
    for i in range(Cdist.shape[0]):
        Cdistances_ids[i,:] = Cdist[i,:].argsort()[:num_hits]
        Cdistances[i,:] = Cdist[i, Cdistances_ids[i,:]]
    
    return Cdistances_ids, Cdistances


##
## ---------------------------- Plotting functions ----------------------------
## 

import matplotlib.pyplot as plt
from dna_features_viewer import GraphicFeature, GraphicRecord


def plot_bgc_genes(BGC_genes, candidate_ids, candidate_dist, 
                   sharex=True, labels=False, dist_method = "centroid"):
    # plot bgc genes for visual comparison
    # 
    max_plot_dimension = 10
    
    num_plots = len(BGC_genes)
    if num_plots > max_plot_dimension:
        print("This might be too many BGCs to compare...")

    # collect all notes and types of the bgcs
    found_types = []
    notes_found = []
    for genes in BGC_genes: 
        for feature in genes:
            found_types.append(feature[3])
            if feature[2] != []:
                note = feature[2].replace(":", " ").split()
                note = [note[1], note[2]]  
                notes_found.append(note)    
    notes_unique = list(set(list(zip(*notes_found))[0]))
    selected_colors = get_spaced_colors(len(notes_unique)+1)                
                
#    fig = plt.figure(figsize=(8, 3.*num_plots))
    fig, ax0 = plt.subplots(len(BGC_genes), 1, figsize=(8, 3.*num_plots) , sharex=sharex) 
    fig.suptitle("Gene feature comparison (similarity measure: " + dist_method + ")")
    max_xlim = max([x[-1][1][1] for x in BGC_genes])
    
    for i, genes in enumerate(BGC_genes):
        record = []
        features = []
        for feature in genes:
            if feature[2] != []:
                color = selected_colors[notes_unique.index(feature[2].replace(":", " ").split()[1])]
            else:
                color = "black"
            
            if labels:
                label = feature[0]
            else:
                label = None
            features.append(GraphicFeature(start=feature[1][0], 
                                           end=feature[1][1], 
                                           strand=feature[1][2], 
                                           color=color , label=label,
                                           thickness=9, linewidth=0.5, fontdict={"size": 9}))
      
        record = GraphicRecord(sequence_length=features[-1].end, features=features)


        record.plot(ax=ax0[i], with_ruler=True)
#        ax0[i].set_title("BGC no. " + str(int(candidates["id"][i])) )
        info1 = "BGC no. %d     " %candidate_ids[i]
        info2 = dist_method + " distance = %.3f" %candidate_dist[i]
        ax0[i].text(0.02,0.75, info1 + info2, size=10, ha="left", transform=ax0[i].transAxes)
        if sharex:
            ax0[i].set_xlim([ax0[i].get_xlim()[0], max_xlim])





