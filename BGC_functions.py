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


class BGC(object):
    """ Class to run word2vec based similarity measure on antiSMASH BGC data.
    Words are here pfam domains.
    Documents are the series of pfam domains.
    """
       
    def __init__(self):
        pass
        
    
    def get_BGC_data(self, path_bgc_data, path_json, results_file = "BGC_collected_data.json", filefilter="*cluster001.gbk", entry = "single"):        
        """ Collect available strains in BGC folder 
        (bit cumbersome...)
        """
        
        dirs = os.listdir(path_bgc_data)
        strains = fnmatch.filter(dirs, filefilter)
        
        if results_file is not None:
            try: 
                self.BGC_data_dict = functions.json_to_dict(path_json + results_file)
                print("BGC json file found and loaded.")
                collect_new_data = False
                
                with open(path_json + results_file[:-4] + "txt", "r") as f:
                    for line in f:
                        line = line.replace('"', '').replace("'", "").replace("[", "").replace("]", "")
                        self.BGC_documents.append(line.split(", "))
                    
            except FileNotFoundError: 
                print("Could not find file ", path_json,  results_file) 
                collect_new_data = True
            
            
                
        # (Maybe) collect data from gbk files:
        if self.BGC_data_dict == {} or results_file is None: 
            collect_new_data = True
            # run over all strains:
            strainnumber = 0
            for bgc_filename in strains:
                bgc_filename_pattern = bgc_filename[0:-7] + "*"
                strainnumber += 1
            
                #BGC_data = GBK_BGC_importer.get_BGC_data(datafolder, "2517287019_c00001_Salpac3...cluster*", "_2.")
                print("collecting data from ...", bgc_filename_pattern)
                BGC_data_dict_strain, BGC_documents_strain, BGC_sequences = BGC_functions.import_BGC_data(path_bgc_data, 
                                                                                           bgc_filename_pattern, "_2.", entry = entry)
                    
                # Make collection of BGC data (includes PFAMS, genes etc.)
                self.BGC_data_dict = {**self.BGC_data_dict, **BGC_data_dict_strain} #BGC_data = BGC_data + BGC_data_strain
                # Make collection of documents (= clusters written in PFAM domains)
                self.BGC_documents = self.BGC_documents + BGC_documents_strain   
                # Make collection of DNA sequences
                self.BGC_sequences = self.BGC_sequences + BGC_sequences
                
                
#                for i in range(0, len(self.BGC_data_dict)):
#                    bgcs_found.append(f"bgc_{strainnumber}_{i}")
#                    bgc_types.append(BGC_data[i][0])
                                        
        # Save collected data
        if collect_new_data == True:
            
            functions.dict_to_json(self.BGC_data_dict, path_json + results_file)     
            # store documents (PFAM domains per BGC)
            with open(path_json + results_file[:-4] + "txt", "w") as f:
                for s in self.BGC_documents:
                    f.write(str(s) +"\n")







def load_BGC_data(datafolder, filename_include, filename_exclude, entry = "single"):
    """ Extract values from antiSMAH .gbk files of byosynthetic gene clusters
    
    """
    
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



