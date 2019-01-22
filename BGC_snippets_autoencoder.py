"""
Test autoencoders to obtain dimensionality reduction of BGC's described in PFAM domains

Jan 2019
Florian Huber
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers


encoding_dim = 100  # Number of dimensions of representation
input_dim = len(bgc_word2vec.dictionary)  # Number of words in corpus


# this is our input placeholder
input_data = Input(shape=(input_dim,))

## "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_data)

# 2 layer network
x = Dense(200, activation='relu')(input_data)
x = Dense(120, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(x)

# add a Dense layer with a L1 activity regularizer
#encoded = Dense(encoding_dim, activation='relu',
#                activity_regularizer=regularizers.l1(10e-5))(input_data)

## "decoded" is the lossy reconstruction of the input
#decoded = Dense(input_dim, activation='sigmoid')(encoded)

x = Dense(120, activation='sigmoid')(encoded)
x = Dense(200, activation='sigmoid')(x)
decoded = Dense(input_dim, activation='sigmoid')(x)


# this model maps an input to its reconstruction
autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

#
## this model maps an input to its encoded representation
encoder = Model(input_data, encoded)
#
#
#
## create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
#
## retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-4]
#
## create the decoder model
#decoder = Model(encoded_input, decoder_layer(encoded_input))
decoder = Model(encoded_input, decoded)


#%% MAKE ONE-HOT word vector
import numpy as np

def one_hot_wv(vocab_size, text_idx):
    # Create one-hot representation from bow word vector
#    vocab_size = len(vocab)
    text_length = len(text_idx)
    one_hot = np.zeros((vocab_size)) #text_length]))
    one_hot[text_idx] = 1
    return one_hot

def full_wv(vocab_size, word_idx, word_count):
    # Create full word vector
    text_length = len(word_idx)
    one_hot = np.zeros((vocab_size)) #text_length]))
    one_hot[word_idx] = word_count
    return one_hot

#word_vector_bow = np.array([x[0] for x in bow_corpus[0]])
#word_vector_count = np.array([x[1] for x in bow_corpus[0]])
#word_vector_1hot = one_hot_wv(len(dictionary), word_vector_bow)
#word_vector_full = full_wv(len(dictionary), word_vector_bow, word_vector_count)
# TODO: not ideal implementation (only testing --> slow!!!)

#%% CREATE DATASET x_train, X-valid, x_test
    

testnum =  len(bow_corpus) 

dict_dim = len(dictionary)
X_data = np.zeros((testnum, dict_dim))

for i, bow_doc in enumerate(bow_corpus[:testnum]):
    word_vector_bow = np.array([x[0] for x in bow_doc]).astype(int)
    word_vector_count = np.array([x[1] for x in bow_doc]).astype(int)
#    X_data[i,:] = one_hot_wv(dict_dim, word_vector_bow)
    X_data[i,:] = full_wv(dict_dim, word_vector_bow, word_vector_count)
    
# shuffling
np.random.shuffle(X_data) 

split_training = [0.8, 0.15, 0.15]

split = [int(testnum *split_training[0]), int(testnum *(split_training[1]+split_training[0]))]

x_train = X_data[:split[0]]
x_test = X_data[split[0]:split[1]]
x_valid = X_data[split[1]:]


#%% DEEP AUTOENCODER
from keras.models import Sequential

encoding_dim = 100  # Number of dimensions of representation
input_dim = len(bgc_word2vec.dictionary)
layer1_factor = 4
layer2_factor = 2

autoencoder = Sequential()

# Encoder Layers
#autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu', activity_regularizer=regularizers.l2(10e-6)))
#autoencoder.add(Dense(2 * encoding_dim, activation='relu', activity_regularizer=regularizers.l2(10e-6)))
autoencoder.add(Dense(layer1_factor * encoding_dim, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(layer2_factor * encoding_dim, activation='relu'))
autoencoder.add(Dense(encoding_dim, activation='relu'))

# Decoder Layers
autoencoder.add(Dense(layer2_factor * encoding_dim, activation='relu'))
autoencoder.add(Dense(layer1_factor * encoding_dim, activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

autoencoder.summary()


# Encoder
input_img = Input(shape=(input_dim,))
encoder_layer1 = autoencoder.layers[0]
encoder_layer2 = autoencoder.layers[1]
encoder_layer3 = autoencoder.layers[2]
encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))

encoder.summary()


#%% DEEP AUTOENCODER 2
from keras.models import Sequential

encoding_dim = 100  # Number of dimensions of representation
input_dim = len(bgc_word2vec.dictionary)
layer1_factor = 8
layer2_factor = 4
layer3_factor = 2

autoencoder = Sequential()

# Encoder Layers
#autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu', activity_regularizer=regularizers.l2(10e-6)))
#autoencoder.add(Dense(2 * encoding_dim, activation='relu', activity_regularizer=regularizers.l2(10e-6)))
autoencoder.add(Dense(layer1_factor * encoding_dim, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(layer2_factor * encoding_dim, activation='relu'))
autoencoder.add(Dense(layer3_factor * encoding_dim, activation='relu'))
autoencoder.add(Dense(encoding_dim, activation='relu'))

# Decoder Layers
autoencoder.add(Dense(layer3_factor * encoding_dim, activation='relu'))
autoencoder.add(Dense(layer2_factor * encoding_dim, activation='relu'))
autoencoder.add(Dense(layer1_factor * encoding_dim, activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

autoencoder.summary()


# Encoder
input_img = Input(shape=(input_dim,))
encoder_layer1 = autoencoder.layers[0]
encoder_layer2 = autoencoder.layers[1]
encoder_layer3 = autoencoder.layers[2]
encoder_layer4 = autoencoder.layers[3]
encoder = Model(input_img, encoder_layer4(encoder_layer3(encoder_layer2(encoder_layer1(input_img)))))

encoder.summary()



#%% CONV AUTOENCODER
from keras.layers import Dense, Conv1D, MaxPooling1D, UpSampling1D

input_dim = len(bgc_word2vec.dictionary)

encoding_dim = 50  # Number of dimensions of representation
layer1_filters = 20
layer2_filter = 10

autoencoder = Sequential()

# Encoder Layers
autoencoder.add(Conv1D(filters=layer1_filters, kernel_size=20, input_shape=(input_dim,1), 
                       kernel_initializer='he_normal', activation='relu', padding='same'))

autoencoder.add(MaxPooling1D(3))
autoencoder.add(Conv1D(filters=layer2_filter, kernel_size=20, kernel_initializer='he_normal', activation='relu'))
autoencoder.add(MaxPooling1D(3))
autoencoder.add(Conv1D(filters=layer2_filter, kernel_size=20, kernel_initializer='he_normal', activation='relu'))
autoencoder.add(Dense(encoding_dim, activation='relu'))

# Decoder Layers
autoencoder.add(Conv1D(filters=layer2_filter, kernel_size=20, kernel_initializer='he_normal', activation='relu'))
autoencoder.add(UpSampling1D(3))
autoencoder.add(Conv1D(filters=layer2_filter, kernel_size=20, kernel_initializer='he_normal', activation='relu'))
autoencoder.add(UpSampling1D(3))
autoencoder.add(Conv1D(filters=layer1_filters, kernel_size=20, kernel_initializer='he_normal', activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

autoencoder.summary()





#%% TRAIN
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=40,
                batch_size=1024,
                validation_data=(x_test, x_test))


#%%
encoded_bgcs = encoder.predict(x_test)

decoded_bgcs = autoencoder.predict(x_test)



#%%

#carefull: use UNSHUFFLED data
testnum =  len(bow_corpus) 

dict_dim = len(dictionary)
X_data_unshuffled = np.zeros((testnum, dict_dim))

for i, bow_doc in enumerate(bow_corpus[:testnum]):
    word_vector_bow = np.array([x[0] for x in bow_doc]).astype(int)
    word_vector_count = np.array([x[1] for x in bow_doc]).astype(int)
    X_data_unshuffled[i,:] = full_wv(dict_dim, word_vector_bow, word_vector_count)



#%%
from scipy import spatial

autoencoder_vectors = encoder.predict(X_data_unshuffled)
Cdist_ae = spatial.distance.cdist(autoencoder_vectors, autoencoder_vectors, 'cosine')

#%%
num_centroid_hits = 25

# Create numpy arrays to store distances
Cdistances_ae_ids = np.zeros((Cdist_ae.shape[0],num_centroid_hits), dtype=int)
Cdistances_ae = np.zeros((Cdist_ae.shape[0],num_centroid_hits))

for i in range(Cdist_ae.shape[0]):
    Cdistances_ae_ids[i,:] = Cdist_ae[i,:].argsort()[:num_centroid_hits]
    Cdistances_ae[i,:] = Cdist_ae[i, Cdistances_ae_ids[i,:]]


#%%
import helper_functions as functions

bgc_id = 43 # intersting ones: 0, 7, 8, 9, 14, 43 (some very similar ones), 58
# interesting also 65, 76
# see 81 --> Short BGCs more quickly seem to be ranked as close!
# see 3108 --> both autoencoder and centroid method find similar structures even if label does not!
num_candidates = 10
select = Cdistances_ae_ids[bgc_id,:num_candidates]
select0 = Cdistances_ae[bgc_id,:num_candidates]

keys = []
values = []
for key, value in bgc_word2vec.BGC_data_dict.items():
    keys.append(key)
    
genes = []
for i, candidate_id in enumerate(select):
    key = keys[candidate_id]
    genes.append(bgc_word2vec.BGC_data_dict[key]["BGC genes"])

functions.plot_bgc_genes(genes, select, select0, sharex=True, labels=False, dist_method = "autoencoder")


select = Cdistances_ids[bgc_id,:num_candidates]
select0 = Cdistances[bgc_id,:num_candidates]

keys = []
values = []
for key, value in bgc_word2vec.BGC_data_dict.items():
    keys.append(key)
    
genes = []
for i, candidate_id in enumerate(select):
    key = keys[candidate_id]
    genes.append(bgc_word2vec.BGC_data_dict[key]["BGC genes"])

functions.plot_bgc_genes(genes, select, select0, sharex=True, labels=False, dist_method = "centroid")



#%%
#%%
# Make network for cytoscape

Bnet = BGC_functions.BGC_distance_network(Cdistances_ae_ids, Cdistances_ae*100,
                                          filename="Bnet_autoencoder_03.graphml", cutoff_dist=0.6)







#%% Compare to PCA and t-SNE
from sklearn.decomposition import PCA



#%%
BGC_names, BGC_types = BGC_functions.BGC_get_types(BGC_data_dict, filename = 'test.csv', strain_lookup_list = None)

def types_to_colors(input_list):
    types = list(set(input_list))
    new_list = [types.index(x) for x in input_list]
    return np.array(new_list)/np.max(np.array(new_list))

type_colors = types_to_colors(BGC_types)
    

#%% PCA from centroid vectors
pca = PCA(n_components=3)
pca_result = pca.fit_transform(centroid_vectors)

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 9))
colors = pca_result[:, 2]
colors = colors - min(colors)
colors = colors/ max(colors)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=type_colors )
plt.show()

#%% PCA from full 1-hot word vectors
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_data_unshuffled)

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 9))
colors = pca_result[:, 1]
colors = colors - min(colors)
colors = colors/ max(colors)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=type_colors )
plt.show()


#%% PCA from full 1-hot word vectors (analog to centroid vectors or autoencoder!)
pca = PCA(n_components=100)
vectors_pca = pca.fit_transform(X_data_unshuffled)

#%%
Cdist_pca = spatial.distance.cdist(vectors_pca, vectors_pca, 'cosine')

num_centroid_hits = 25

# Create numpy arrays to store distances
Cdistances_pca_ids = np.zeros((Cdist_pca.shape[0],num_centroid_hits), dtype=int)
Cdistances_pca = np.zeros((Cdist_pca.shape[0],num_centroid_hits))

for i in range(Cdist_pca.shape[0]):
    Cdistances_pca_ids[i,:] = Cdist_pca[i,:].argsort()[:num_centroid_hits]
    Cdistances_pca[i,:] = Cdist_pca[i, Cdistances_pca_ids[i,:]]


#%%
import helper_functions as functions

bgc_id = 9 # intersting ones: 0, 7, 8, 9, 14, 43 (some very similar ones), 58
# interesting also 65, 76
# see 81 --> Short BGCs more quickly seem to be ranked as close!
# see 3108 --> both autoencoder and centroid method find similar structures even if label does not!
num_candidates = 10
select = Cdistances_ae_ids[bgc_id,:num_candidates]
select0 = Cdistances_ae[bgc_id,:num_candidates]

keys = []
values = []
for key, value in bgc_word2vec.BGC_data_dict.items():
    keys.append(key)
    
genes = []
for i, candidate_id in enumerate(select):
    key = keys[candidate_id]
    genes.append(bgc_word2vec.BGC_data_dict[key]["BGC genes"])

functions.plot_bgc_genes(genes, select, select0, sharex=True, labels=False, dist_method = "autoencoder")

# Centroid vector distance
select = Cdistances_ids[bgc_id,:num_candidates]
select0 = Cdistances[bgc_id,:num_candidates]

keys = []
values = []
for key, value in bgc_word2vec.BGC_data_dict.items():
    keys.append(key)
    
genes = []
for i, candidate_id in enumerate(select):
    key = keys[candidate_id]
    genes.append(bgc_word2vec.BGC_data_dict[key]["BGC genes"])

functions.plot_bgc_genes(genes, select, select0, sharex=True, labels=False, dist_method = "centroid")

# PCA vector distance
select = Cdistances_pca_ids[bgc_id,:num_candidates]
select0 = Cdistances_pca[bgc_id,:num_candidates]

keys = []
values = []
for key, value in bgc_word2vec.BGC_data_dict.items():
    keys.append(key)
    
genes = []
for i, candidate_id in enumerate(select):
    key = keys[candidate_id]
    genes.append(bgc_word2vec.BGC_data_dict[key]["BGC genes"])

functions.plot_bgc_genes(genes, select, select0, sharex=True, labels=False, dist_method = "PCA")


#%%
# Make network for cytoscape

Bnet = BGC_functions.BGC_distance_network(Cdistances_pca_ids, Cdistances_pca,
                                          filename="Bnet_pca_01.graphml", cutoff_dist=0.08)




#%%

# this model maps an input to its reconstruction
autoencoder = Model(input_data, decoded)

#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#autoencoder.compile(optimizer='adadelta', loss='MAPE')


#%% TRAIN AUTOENCODER
split_training = [0.6, 0.2, 0.2]

split = [int(testnum *split_training[0]), int(testnum *(split_training[1]+split_training[0]))]

x_train = X_data[:split[0]]
x_test = X_data[split[0]:split[1]]
x_valid = X_data[split[1]:]

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=512,
                shuffle=True,
                validation_data=(x_test, x_test))


#%%
encoded_bgcs = encoder.predict(x_test)

decoded_bgcs = decoder.predict(encoded_bgcs)


#%%
encoded_bgcs_rel = encoded_bgcs- np.tile(np.mean(encoded_bgcs, axis = 0), (encoded_bgcs.shape[0], 1))


#%% Display 2D data
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.scatter(encoded_bgcs_rel[:, 0], encoded_bgcs_rel[:, 1])
plt.show()




#%% VARIATIONAL AUTOENCODER
from keras.layers import Input, Dense, Lambda, Layer
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K
from keras import metrics

batch_size = 256
original_dim = len(bgc_word2vec.dictionary)
latent_dim = 200
intermediate_dim = 1000
epochs = 50
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(x)

loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [loss_layer])
vae.compile(optimizer='rmsprop', loss=[zero_loss])


##checkpoint
#cp = [callbacks.ModelCheckpoint(filepath="/home/ubuntu/pynb/model.h5", verbose=1, save_best_only=True)]

#train
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))#, callbacks=cp)

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)


#%%
import csv

#csv.register_dialect('myDialect', delimiter = ';', lineterminator = '\r\n\r\n')
path_data = "C:\\Users\\FlorianHuber\\OneDrive - Netherlands eScience Center\\Project_Wageningen_iOMEGA\\docnade\\data\\BGC_test\\"


split_training = [0.6, 0.2, 0.2]
corpus_size = len(BGC_corpus)
split_training = [int(corpus_size*split_training[0]), int(corpus_size*(split_training[1]+split_training[0]))]
                     
#%%
filename = path_data + "training.csv"
with open(filename, 'w') as csvFile:
    writer = csv.writer(csvFile)
    for i, row in enumerate(BGC_corpus[:split_training[0]]):
        writer.writerow(row)

csvFile.close()

filename = path_data + "validation.csv"
with open(filename, 'w') as csvFile:
    writer = csv.writer(csvFile)
    for i, row in enumerate(BGC_corpus[split_training[0]:split_training[1]]):
        writer.writerow(row)

csvFile.close()

filename = path_data + "test.csv"
with open(filename, 'w') as csvFile:
    writer = csv.writer(csvFile)
    for i, row in enumerate(BGC_corpus[split_training[1]:]):
        writer.writerow(row)

csvFile.close()

#%%
with open(path_data + "bgc_corpus.vocab", "w") as f:
    for i in range(len(dictionary)):
        f.write(dictionary[i] +"\n")
        
#%%
test = []
for i in range(len(dictionary)):
    test.append(dictionary[i])


        
#%%
#cmd = python preprocess.py --input "C:\Users\\FlorianHuber\OneDrive - Netherlands eScience Center\Project_Wageningen_iOMEGA\docnade\data\BGC_test" --output "C:\Users\FlorianHuber\OneDrive - Netherlands eScience Center\Project_Wageningen_iOMEGA\docnade\data" --vocab "C:\Users\\FlorianHuber\OneDrive - Netherlands eScience Center\Project_Wageningen_iOMEGA\docnade\data\BGC_test"
    