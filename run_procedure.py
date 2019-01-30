# data locations
DATASET = "C:\\OneDrive - Netherlands eScience Center\\Project_Wageningen_iOMEGA"
PATH_MS2LDA = DATASET + "\\lda\\code\\"
PATH_MGF_DATA = DATASET + "\\Data\\Crusemann_dataset\\Crusemann_only_Clutered_Data\\"
MIBIG_JSON_DIR = DATASET + "\\Data\\mibig\\mibig_json_1.4"

NODES_FILE = PATH_MGF_DATA + "clusterinfosummarygroup_attributes_withIDs\\0d51c5b6c73b489185a5503d319977ab..out"

MGF_FILE = PATH_MGF_DATA + "METABOLOMICS-SNETS-c36f90ba-download_clustered_spectra-main.mgf"
EDGES_FILE = PATH_MGF_DATA + 'networkedges_selfloop\\9a93d720f69143bb9f971db39b5d2ba2.pairsinfo'

ROOT_PATH = DATASET + "\\Data\mibig_select\\"
FOLDERS = ['NRPS','Others','PKSI','PKS-NRP_Hybrids','PKSother','RiPPs','Saccharides','Terpene']
ANTISMASH_DIR = DATASET +"\\Data\\Crusemann_dataset\\bgc_crusemann\\"

from nplinker_constants import nplinker_setup
nplinker_setup(LDA_PATH=PATH_MS2LDA)

# import from NPlinker
from metabolomics import load_spectra
from metabolomics import load_metadata
from metabolomics import load_edges
#from metabolomics import make_families
from genomics import loadBGC_from_cluster_files
from genomics import make_mibig_bgc_dict

# import general packages
import os
import glob


import helper_functions as functions
import MS_functions

#%%
# load, initialize data
nplinker_setup(LDA_PATH=PATH_MS2LDA)

spectra = load_spectra(MGF_FILE)
load_edges(spectra, EDGES_FILE)
#families = make_families(spectra)
metadata = load_metadata(spectra, NODES_FILE)

input_files = []
ann_files = []
mibig_bgc_dict = make_mibig_bgc_dict(MIBIG_JSON_DIR)

for folder in FOLDERS:
    fam_file = os.path.join(ROOT_PATH, folder)
    cluster_file = glob.glob(fam_file + os.sep + folder + "_clustering*")
    annotation_files = glob.glob(fam_file + os.sep + "Network_*")
    input_files.append(cluster_file[0])
    ann_files.append(annotation_files[0])
gcf_list, bgc_list, strain_list = loadBGC_from_cluster_files(input_files, ann_files, antismash_dir=ANTISMASH_DIR, antismash_format = 'flat', mibig_bgc_dict=mibig_bgc_dict)



#%% Create Corpus!
print(20 * "--")
print("// check issue with losses! So far in metabolomics.py losses were between MS2 peaks?")
print("// ")
print("// Now its between MS1 peak (precursor) and MS2 peaks (better?)")
print(20 * "--")


# quite a large tollerance!!!
MS_documents, MS_documents_intensity = MS_functions.create_MS_documents(spectra, 2)

##%% get feeling for peak range and error:
#peaks_total = []
#for i, spectrum in enumerate(spectra):
#    for peak in spectrum.peaks:
#        peaks_total.append(peak[0])
#
#peaks_total = np.array(peaks_total)
#
#
##%%
#from matplotlib import pyplot as plt
#plt.hist(peaks_total)


#%% cleaned up attempt ---------------------------------------------------------------

from Similarities import SimilarityMeasures

Sim_measure = SimilarityMeasures(MS_documents)


#%%
Sim_measure.preprocess_documents(0.2, create_stopwords = False)


#%%
file_model_word2vec = 'data\\MS_model_word2vec_ref.model' 
Sim_measure.build_model_word2vec(file_model_word2vec, size=100, window=50, 
                             min_count=1, workers=4, iter=25, 
                             use_stored_model=True)

#%%
Sim_measure.get_vectors_centroid(weighted=True)


#%%
Sim_measure.get_centroid_distances(num_hits=25, method='cosine')


#%%
file_model_ae = 'data\\model_autoencoder_test1.h5'
file_model_encoder = 'data\\model_encoder_test1.h5'
Sim_measure.build_autoencoder(file_model_ae, file_model_encoder, epochs = 10, batch_size = 1024, encoding_dim = 100)

#%%
Sim_measure.get_autoencoder_distances(num_hits=25, method='cosine')


#%%
MSnet = functions.create_distance_network(Sim_measure.Cdistances_ae_ids, 
                                Sim_measure.Cdistances_ae * 100000, 
                                filename="data\\MS_ae_test.graphml",                             
                                cutoff_dist = 0.1,
                                max_connections = 10,
                                min_connections = 1)

#%%
MSnet = functions.create_distance_network(Sim_measure.Cdistances_ids, 
                                Sim_measure.Cdistances, 
                                filename="data\\MS_centroid_test.graphml",                             
                                cutoff_dist = 0.2,
                                max_connections = 10,
                                min_connections = 1)


#%%
Sim_measure.get_pca_distances(num_hits=25, method='cosine')

#%%
MSnet = functions.create_distance_network(Sim_measure.Cdistances_pca_ids, 
                                Sim_measure.Cdistances_pca, 
                                filename="data\\MS_pca_test.graphml",                             
                                cutoff_dist = 0.12,
                                max_connections = 10,
                                min_connections = 1)



#%% LDA -----------------------------------------------------------------------
file_model_lda = "data\\model_lda_MS_test01.model"
Sim_measure.build_model_lda(file_model_lda, num_of_topics=100, num_pass=4, 
                        num_iter=100, use_stored_model=True)



#%% 
Sim_measure.get_lda_distances(num_hits=25)

#%%
from matplotlib import pyplot as plt
plt.hist(Sim_measure.Cdistances_lda.reshape(len(Cdist)*25), 50)



#%%
MSnet = functions.create_distance_network(Sim_measure.Cdistances_lda_idx, 
                                Sim_measure.Cdistances_lda, 
                                filename="output\\MS_lda_test.graphml",                             
                                cutoff_dist = 0.15,
                                max_connections = 10,
                                min_connections = 1)

#%% LSI -----------------------------------------------------------------------
file_model_lsi = "data\\model_lsi_MS_test01.model"
Sim_measure.build_model_lsi(file_model_lsi, num_of_topics=100, 
                             use_stored_model=True)


#%% 
Sim_measure.get_lsi_distances(num_hits=25)



#%%
MSnet = functions.create_distance_network(Sim_measure.Cdistances_lsi_idx, 
                                Sim_measure.Cdistances_lsi, 
                                filename="output\\MS_lsi_test.graphml",                             
                                cutoff_dist = 0.1,
                                max_connections = 10,
                                min_connections = 1)


#%% Doc2Vec
file_model_doc2vec = "data\\model_doc2vec_MS_test01.model"
Sim_measure.build_model_doc2vec(file_model_doc2vec, vector_size=100, window=50, 
                             min_count=1, workers=4, epochs=250, 
                             use_stored_model=True)


#%%
Sim_measure.get_doc2vec_distances(num_hits=25, method='cosine')








#%%
candidates = Sim_measure.similarity_search(num_centroid_hits=100, centroid_min=0.3, 
                          doc2vec_refiner = True,
                          WMD_refiner = False)

#%%
import pandas as pd
candidates_pd = []
for distances in candidates:
    distances2 = pd.DataFrame(distances, columns=["id", "centroid", "lda", "LSI", "wmd", "doc2vec"])
    candidates_pd.append(distances2) 

