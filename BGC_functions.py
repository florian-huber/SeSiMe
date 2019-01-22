# -*- coding: utf-8 -*-
"""
Just testing how to import and read a gbk file for the BGCs (biosynthetic gene clusters)
"""
from __future__ import print_function

from Bio import SeqIO
import re
import os, fnmatch
import csv
from difflib import get_close_matches 
import gensim 
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
import numpy as np
from pprint import pprint  # pretty-printer
    
import helper_functions as functions


def import_BGC_data(datafolder, filename_include, filename_exclude, entry = "single"):
    # FUNCTION to extract values from antiSMAH .gbk files of byosynthetic gene clusters
    
    pattern = filename_include
    dirs = os.listdir(datafolder)
    
#    BGC_data = []
    BGC_data_dict = {}
    BGC_documents = []
    BGC_sequences = []
    feature_types = []
    
    
    # go through all the clusters in one strain:
    for bgc_filename in [x for x in fnmatch.filter(dirs, pattern) if filename_exclude not in x ] :

        bgc_file = datafolder + bgc_filename
        
        if entry == "multiple":
            bgc_records = list(SeqIO.parse(bgc_file, "genbank"))
        else:    
#            bgc_records = SeqIO.read(bgc_file, "genbank")
            bgc_records = list(SeqIO.parse(bgc_file, "genbank"))
        
        for num_rec, bgc_record in enumerate(bgc_records):
            if entry == "multiple":
                bgc_filename_updated = bgc_filename + str(num_rec)
            else:    
                bgc_filename_updated = bgc_filename 
            
            bgc_sequence = bgc_record.seq._data
            
            # collect relevant data (or what we believe might become relevant)
            PFAM_domain_data = []   
            PFAM_domains = []     
            feature_types =[] 
            bgc_knownclusters = [] 
            genes = []
             
            # go through all features and look for the most relevant ones
            for i, feature in enumerate(bgc_record.features):
                feature_types.append(feature.type)
                
                if "product" in bgc_record.features[i].qualifiers: 
                    bgc_info = {}
                    bgc_info["BGC type"] = bgc_record.features[i].qualifiers["product"][0]
                    if "probability" in bgc_record.features[i].qualifiers: 
                        bgc_info["BGC type probability"] = bgc_record.features[i].qualifiers["probability"][0]
                    else:
                        bgc_info["BGC type probability"] = 0
                    
                if "knownclusterblast" in bgc_record.features[i].qualifiers:    
                    for m in range(0,len(bgc_record.features[i].qualifiers["knownclusterblast"])):
                            
                        teststring = bgc_record.features[i].qualifiers["knownclusterblast"][m]
                        bgc_knownclusters.append([teststring.split("\t")[0][teststring.find("B"):],
                        [float(s) for s in re.findall(r'-?\d+\.?\d*', teststring.split("\t")[1])][-1]
                        ])
                           
                # collect key genes (= CDS only?):
                if feature.type == "CDS":
                    location = bgc_record.features[i].location
                    features = []
                    features.append(bgc_record.features[i].qualifiers["locus_tag"][0])
                    features.append([location.nofuzzy_start, location.nofuzzy_end, location._strand],)
                    if "note" in bgc_record.features[i].qualifiers: 
                        features.append(bgc_record.features[i].qualifiers["note"][0])
                    else:
                        features.append([])
                    if "sec_met" in bgc_record.features[i].qualifiers:
                        features.append(bgc_record.features[i].qualifiers["sec_met"][0])
                    else:
                        features.append([])
    #                         bgc_record.features[i].qualifiers["translation"][0]
                        
                    genes.append(features)
        
                # collect PFAM domains (and antiSMASH scores):
                if feature.type == "PFAM_domain":
#                    if "db_xref" in feature.qualifiers:
                    PFAM_domains.append(feature.qualifiers['db_xref'][0][6:])
                    PFAM_domain_data.append([feature.qualifiers['db_xref'][0][6:],
                                           feature.qualifiers["evalue"][0],
                                           feature.qualifiers["score"][0],
                                           float(feature.qualifiers["note"][1][27:])])
                      
            BGC_data_dict[bgc_filename_updated] = {"BGC type" : bgc_info["BGC type"],
                         "BGC type probability" : bgc_info["BGC type probability"],
                         "similar known BGCs": bgc_knownclusters,
                         "PFAM domains" : PFAM_domain_data,
                         "BGC genes" : genes}
            BGC_documents.append(PFAM_domains)
            BGC_sequences.append(bgc_sequence)
    return BGC_data_dict, BGC_documents, BGC_sequences 


def BGC_lda_model(BGC_documents, num_of_topics=100, num_pass=1, num_iter=50):
    
    # first step: use LDA model for closest hits...
    BGC_texts, frequency = PFAM_dictionary(BGC_documents)

    #BGC_corpus = Dictionary(BGC_texts)
    dictionary = corpora.Dictionary(BGC_texts)
    corpus = [dictionary.doc2bow(text) for text in BGC_texts]
    
    lda_model = gensim.models.LdaModel(corpus, id2word=dictionary, 
                                       num_topics=num_of_topics, passes=num_pass, iterations=num_iter) 
    
    #LdaMulticore(corpus=BGC_corpus, num_topics=100)
    # Output the Keyword in the 10 topics
    pprint("Keyword in the 10 topics")
    pprint(lda_model.print_topics())

    return BGC_texts, dictionary, corpus, lda_model
      

def find_all_nearest_connections(BGC_documents, 
                          BGC_texts,
                          dictionary,
                          corpus,
                          model_word2vec, 
                          lda_model,
                          num_lda_hits=50, 
                          stopwords = [],
                          WMD_refiner = False):
    # Search nearest neighbors in two steps:
    # 1- Use cosine measure based on LDA model to select candidates
    # 2- Calculate distances more accurately for the the top candidates using WMD (slow!)
    
    if num_lda_hits > len(BGC_documents):
        num_lda_hits = len(BGC_documents)
        print('number of best lda hits to keep is bigger than given dimension.')
    
    try: lda_model
    except NameError: lda_model = None
    if lda_model is None:
        print("No lda model found. Calculate new one...")
        BGC_texts, dictionary, corpus, lda_model = BGC_lda_model(BGC_documents, num_pass=5, num_iter=100)
    else:
        print("Lda model found")
    
    index = gensim.similarities.MatrixSimilarity(lda_model[corpus])
    
    # Calulate distances based on LDA topics...
    dimension = len(BGC_documents)
    distances = np.zeros((dimension, dimension))
        
    keeptrack = []
    # TODO: Parallize the following part of code!!
    # TODO: try smarter way of finding num_lda_hits based on random testing first and then a cutoff 
    for i, documents in enumerate(BGC_texts):
        # calculate distances between BGCs
        print('\r', 'Document ', i, ' of ', len(BGC_texts), end="")
         
        query = [word.lower() for word in documents if word not in stopwords] 
        vec_bow = dictionary.doc2bow(query)
        vec_lda = lda_model[vec_bow]
        list_similars = index[vec_lda]
        list_similars = sorted(enumerate(list_similars), key=lambda item: -item[1])
        
        if WMD_refiner:  
            candidates = [x[0] for x in list_similars[:num_lda_hits] if not x[0] == i]            
            for m in candidates:
                if (i, m) not in keeptrack: # avoid double work
                    distances[i,m] = model_word2vec.wmdistance(BGC_documents[i], BGC_documents[m])
                    distances[m,i] = distances[i,m]
                    keeptrack.append((i,m))
                    keeptrack.append((m,i))
        else:
            candidates = [x[0] for x in list_similars[:num_lda_hits] if not x[0] == i]
            candidate_dist = [x[1] for x in list_similars[:num_lda_hits] if not x[0] == i]
            for m, neighbor in enumerate(candidates):       
                distances[i,neighbor] = candidate_dist[m]
                distances[neighbor,i] = candidate_dist[m]
        
    distances[distances > 1E10] = 0  # remove infinity values        
    return distances


def find_nearest_connections(BGC_query,
                          BGC_documents,
                          BGC_texts,
                          dictionary,
                          corpus,
                          model_word2vec, 
                          lda_model,
                          lda_index = None,
                          num_lda_hits=50, 
                          lda_min=0.9995,
                          stopwords = [],
                          WMD_refiner = False):
    # Search nearest neighbors of one BGC, in two steps:
    # 1- Use cosine measure based on LDA model to select candidates
    # 2- Calculate distances more accurately for the the top candidates using WMD (slow!)
    
    if num_lda_hits > len(BGC_documents):
        num_lda_hits = len(BGC_documents)
        print('number of best lda hits to keep is bigger than given dimension.')
    
    try: lda_model
    except NameError: lda_model = None
    if lda_model is None:
        print("No lda model found. Calculate new one...")
        BGC_texts, dictionary, corpus, lda_model = BGC_lda_model(BGC_documents, num_pass=5, num_iter=100)
    else:
        print("Lda model found")
    
    try: lda_index
    except NameError: lda_index = None
    if lda_index is None:
        print("No lda index found. Calculate new one...")
        lda_index = gensim.similarities.MatrixSimilarity(lda_model[corpus])
    else:
        print("Lda index found")
    
    # Calulate distances based on LDA topics...            
    query = [word.lower() for word in BGC_query if word not in stopwords] 
    vec_bow = dictionary.doc2bow(query)
    vec_lda = lda_model[vec_bow]
    list_similars = lda_index[vec_lda]
    list_similars = sorted(enumerate(list_similars), key=lambda item: -item[1])
    list_similars_np = np.array(list_similars)
    # check that at least all very high (>lda_min) values are included 
    if np.where(list_similars_np[:,1] > lda_min)[0].shape[0] > num_lda_hits: 
        updated_hits = np.where(list_similars_np[:,1] > lda_min)[0].shape[0]
        candidates = [x[0] for x in list_similars[:updated_hits]]
        distances = np.zeros((updated_hits, 3))
        distances[:,0] = np.array(candidates) 
        distances[:,1] = np.array([x[1] for x in list_similars[:updated_hits]])      
    else:
        candidates = [x[0] for x in list_similars[:num_lda_hits]]
        distances = np.zeros((num_lda_hits, 3))
        distances[:,0] = np.array(candidates) 
        distances[:,1] = np.array([x[1] for x in list_similars[:num_lda_hits]]) 
        
    if WMD_refiner:  
        for i, m in enumerate(candidates):
            distances[i,2] = model_word2vec.wmdistance(BGC_query, BGC_documents[m])
            
    distances[distances[:,2] > 1E10,2] = np.max(distances[:,2]) # remove infinity values    
    distances = distances[np.lexsort((distances[:,1], distances[:,2])),:]    
    return distances, lda_index


def BGC_distance_network(Cdistances_ids, Cdistances, filename="Bnet_word2vec_test.graphml", cutoff_dist=0.15):
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
        new_edges = [(i, int(Cdistances_ids[i,x]), float(Cdistances[i,x])) for x in idx if Cdistances_ids[i,x] != i]
        Bnet.add_weighted_edges_from(new_edges)
#        Bnet.add_edge(i, int(candidate), weight=float((max_distance - distances[i,candidate])/max_distance) )
        
    # export graph for drawing (e.g. using Cytoscape)
    nx.write_graphml(Bnet, filename)
    return Bnet


def BGC_get_types(BGC_data_dict, filename = 'Bnet_clusterlabels.csv', strain_lookup_list = None):
    # get BGC names for network nodes (e.g. to use in Cytoscape)    
    
    BGC_names = []
    BGC_names_v2 = []
    BGC_types = []
    BGC_filename = []
    
    strain_count = 0
    bgc_count = 0
    strain_name = ""
    strain_name_new = ""
    
    if strain_lookup_list is not None:
        # read input csv file with strain name lookup table:
        with open(strain_lookup_list, newline='') as csvfile:
            csv_input = csv.reader(csvfile, delimiter=',', quotechar='|')
            strainnames = []
            for row in csv_input:
                strainnames.append(row)
        
        # assume that first row is old name tag and second row is the aimed name
        strainnames_bgc= [x[0] for x in strainnames]  
            
    for key, value in BGC_data_dict.items():
        bgc_count += 1
        strain_name_new = key[:-15]
        if strain_name != strain_name_new:
            strain_name = strain_name_new
            strain_count +=1
            bgc_count = 1   
        
        if strain_lookup_list is None:
            BGC_names.append(strain_name_new + "_bgc_" + str(bgc_count))
        else:
            tag = get_close_matches(strain_name_new, strainnames_bgc, n=1, cutoff=0.01)
            BGC_names.append(strainnames[strainnames_bgc.index(tag[0])][1] + "_bgc_" + str(bgc_count))    
        
        BGC_names_v2 .append(str(strain_count) + "_bgc_" + str(bgc_count) )
        BGC_types.append(value["BGC type"])   
        BGC_filename.append(key)
    
    # export additional information as table
    csv.register_dialect('myDialect', delimiter = ';', lineterminator = '\r\n\r\n')
    
    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["node;" + "cluster name;"  + "cluster name alternative;" 
                         + "cluster type;" + "bgc filename;"])
        for i, row in enumerate(BGC_types):
            writer.writerow([str(i) + ";" + BGC_names[i] + ";" + BGC_names_v2[i] + ";" + BGC_types[i] + ";" + BGC_filename[i] + ";"])
    
    csvFile.close()

    return BGC_names, BGC_types



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


def create_centroid_vectors_old(corpus, idf_scores, model_word2vec, 
                            extra_stopwords=[], weighted="tfidf"):
    """ Calculate centroid vectors for all documents in corpus
    
     centroid vector = sum over all word vectors (except words from extra_stopwords)
     if weighted == "tfidf" --> will be weighted by the tf*idf score
     if weighted == "idf" --> will be weighted by the idf score
     
     Args:
     --------
     corpus: 
         Word corpus
     idf_scores:
         
     model_word2vec
     extra_stopwords
     weighted
     """
    word_vector_length = model_word2vec.vector_size
    centroid_vectors = []
    corpus_size = len(corpus)
    
    for n, document in enumerate(corpus):
        if (n+1) % 100 == 0 or n == corpus_size-1:  # show progress
                print('\r', ' Calculated centroid vectors for ', n+1, ' of ', corpus_size, ' documents.', end="")
            
        word_weight = np.zeros(len(document))
        word_vector = np.zeros(word_vector_length)
        if weighted == "idf":
            tf = 1/(max(1, len(document)))  # term frequency
        else:
            tf = 1
            
        gen = ((i, word) for (i, word) in enumerate(document) if word not in extra_stopwords)
        for i, word in gen:
            if weighted:
                word_weight[i] = (tf * idf_scores["idf score"][idf_scores["word"] == word]).iloc[0]
                word_vector = word_vector + word_weight[i]*model_word2vec.wv[word]
            else:
                word_vector = word_vector + model_word2vec.wv[word]
                
        if weighted:    
            word_vector = word_vector/np.sum(word_weight)
        centroid_vectors.append(word_vector)
    
