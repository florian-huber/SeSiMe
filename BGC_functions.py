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
import numpy as np
    
import helper_functions as functions


class BGC(object):
    """ Class to run word2vec based similarity measure on antiSMASH BGC data.
    Words are here pfam domains.
    Documents are the series of pfam domains.
    """
       
    def __init__(self):
        self.id = []

    
    def read_BGC_data(self, bgc_record, bgc_filename_updated, id):        
        """ Read .gbk file and extract most relevant information
        
        """
        
        bgc_sequence = bgc_record.seq._data
            
        # Collect relevant data (or what we believe might become relevant)
        PFAM_domain_data = []   
        PFAM_domains = []     
        feature_types =[] 
        bgc_knownclusters = [] 
        genes = []
        bgc_info = {}
         
        # Go through all features and look for the most relevant ones
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
                if "location" in bgc_record.features[i].qualifiers:
                    features.append([location.nofuzzy_start, location.nofuzzy_end, location._strand],)
                else:
                    features.append([])
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
        
        self.id = id
        if "BGC type" not in bgc_info:
            bgc_info["BGC type"] = "unkown"
            bgc_info["BGC type probability"] = "unkown"
            print("Missing feature: bgc type." )
        self.bgc_type = (bgc_info["BGC type"], bgc_info["BGC type probability"])
        self.pfam_domains = PFAM_domains
        self.pfam_domain_data = PFAM_domain_data
        self.genes = genes
        self.sequences = bgc_sequence
        self.bgc_knownclusters = bgc_knownclusters
        



def load_BGC_data(path_bgc_data, 
                  filename_include, 
                  filename_exclude, 
                  path_json, 
                  results_file = "BGC_collected_data.json", 
                  filefilter="*cluster001.gbk",
                  remove_for_small_files = 0,
                  entry = "single"):
    """ Extract values from antiSMAH .gbk files of byosynthetic gene clusters
    
    """
    
    BGCs = []
    BGCs_dict = {}
    BGC_documents = []

#    dirs = os.listdir(path_bgc_data)
#    dirs_filtered = [x for x in fnmatch.filter(dirs, filename_include) if filename_exclude not in x]
#    strains = fnmatch.filter(dirs_filtered, filefilter)
    
    strains = []
    list_of_files = []
    for (dirpath, dirnames, filenames) in os.walk(path_bgc_data):
        if len(filenames) > 0:
            list_of_files.append(filenames[0])
        if (len(filenames) > 0) & (len(fnmatch.filter(filenames, filefilter)) > 0):
            strains.append((dirpath, filenames[0]))

    if results_file is not None:
        try: 
            BGCs_dict = functions.json_to_dict(path_json + results_file)
            print("BGC json file found and loaded.")
            collect_new_data = False
            
            with open(path_json + results_file[:-4] + "txt", "r") as f:
                for line in f:
                    line = line.replace('"', '').replace("'", "").replace("[", "").replace("]", "").replace("\n", "")
                    BGC_documents.append(line.split(", "))
                
        except FileNotFoundError: 
            print("Could not find file ", path_json,  results_file) 
            collect_new_data = True

    # Read data from files if no pre-stored data is found:
    if BGCs_dict == {} or results_file is None: 
        collect_new_data = True
        bgc_count = 0

        # Run over all strains:
        strainnumber = 0
        for bgc_path, bgc_filename in strains:
            if remove_for_small_files > 0:
                bgc_filename_pattern = bgc_filename[0:-remove_for_small_files] + "*"
            else: 
                bgc_filename_pattern = bgc_filename
            strainnumber += 1
        
            print("collecting data from ...", bgc_filename_pattern)
            
            # Go through all clusters in one strain:
#            for bgc_filename in [x for x in fnmatch.filter(dirs, bgc_filename_pattern) if filename_exclude not in x]:     
            for bgc_filename in [x for x in fnmatch.filter(list_of_files, bgc_filename_pattern) if filename_exclude not in x]:    
                bgc_file = os.path.join(bgc_path, bgc_filename)
                
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
                    
                    bgc = BGC()
                    bgc.read_BGC_data(bgc_record, bgc_filename_updated, bgc_count)
                    bgc_count += 1 
            
                # Collect in form of list of spectrum objects, and as dictionary
                BGCs.append(bgc)
                BGCs_dict[bgc_filename_updated] = bgc.__dict__
                BGC_documents.append(bgc.pfam_domains)
                                    
    # Save collected data
    if collect_new_data == True:
        
        functions.dict_to_json(BGCs_dict, path_json + results_file)     
        # Store documents (PFAM domains per BGC)
        with open(path_json + results_file[:-4] + "txt", "w") as f:
            for s in BGC_documents:
                f.write(str(s) +"\n")
    
    
    return BGCs, BGCs_dict, BGC_documents
    



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
    """ Get BGC names for network nodes (e.g. to use in Cytoscape)    
    
    """
    
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


##
## ---------------------------- Plotting functions ----------------------------
## 

import matplotlib.pyplot as plt
from dna_features_viewer import GraphicFeature, GraphicRecord




def get_spaced_colors(n):
    """ Create set of 'n' well-distinguishable colors
    """
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    RGB_colors = [(int(i[:2], 16)/255, int(i[2:4], 16)/255, int(i[4:], 16)/255) for i in colors]    

    return RGB_colors



def plot_bgc_genes(query_id, BGCs_dict, BGC_measure, num_candidates = 10, 
                   sharex=True, labels=False, dist_method = "centroid",
                   spacing = 1):
    """ Plot bgc genes for visual comparison
    
    """ 

    # Select chosen distance methods
    if dist_method == "centroid":
        candidates_idx = BGC_measure.Cdistances_ctr_idx[query_id, :num_candidates]
        candidates_dist = BGC_measure.Cdistances_ctr[query_id, :num_candidates]
    elif dist_method == "pca":
        candidates_idx = BGC_measure.Cdistances_pca_idx[query_id, :num_candidates]
        candidates_dist = BGC_measure.Cdistances_pca[query_id, :num_candidates]
    elif dist_method == "autoencoder":
        candidates_idx = BGC_measure.Cdistances_ae_idx[query_id, :num_candidates]
        candidates_dist = BGC_measure.Cdistances_ae[query_id, :num_candidates]
    elif dist_method == "lda":
        candidates_idx = BGC_measure.Cdistances_lda_idx[query_id, :num_candidates]
        candidates_dist = BGC_measure.Cdistances_lda[query_id, :num_candidates]
    elif dist_method == "lsi":
        candidates_idx = BGC_measure.Cdistances_lsi_idx[query_id, :num_candidates]
        candidates_dist = BGC_measure.Cdistances_lsi[query_id, :num_candidates]
    elif dist_method == "doc2vec":
        candidates_idx = BGC_measure.Cdistances_d2v_idx[query_id, :num_candidates]
        candidates_dist = BGC_measure.Cdistances_d2v[query_id, :num_candidates]
    else:
        print("Chosen distance measuring method not found.")


    keys = []
    for key, value in BGCs_dict.items():
        keys.append(key)  
        
    BGC_genes = []  
    for i, candidate_id in enumerate(candidates_idx):
        key = keys[candidate_id]
        BGC_genes.append(BGCs_dict[key]["genes"])

    # Collect all notes and types of the bgcs
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
    fig, ax0 = plt.subplots(len(BGC_genes), 1, figsize=(10, spacing*num_candidates) , sharex=sharex) 
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
        info1 = "BGC no. %d     " %candidates_idx[i]
        info2 = dist_method + " distance = %.3f" %candidates_dist[i]
        ax0[i].text(0.02,0.75, info1 + info2, size=10, ha="left", transform=ax0[i].transAxes)
        if sharex:
            ax0[i].set_xlim([ax0[i].get_xlim()[0], max_xlim])

