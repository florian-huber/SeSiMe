from __future__ import print_function

import os
import numpy as np
import logging
from pprint import pprint 
import gensim 
from gensim import corpora
from gensim import models
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

from scipy import spatial
from sklearn.decomposition import PCA

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense

import helper_functions as functions


       
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training progress.
    Used to keep track of gensim model training (word2vec, lda...)'''
    def __init__(self, num_of_epochs):
        self.epoch = 0
        self.num_of_epochs = num_of_epochs
    def on_epoch_end(self, model):
        print('\r', 'Epoch ', (self.epoch+1), ' of ', self.num_of_epochs, '.' , end="")
        self.epoch += 1


class SimilarityMeasures():
    """ Class to run different similarity measure on sentence-like data.
    Words can be representing all kind of things (Pfam domains for proteins, peaks for spectra etc.).
    Documents lists of words.
    
    Similarity measuring methods:
    1) Low-dimensional document vector similarity (e.g. cosine similarity)
        a) Word2Vec based centroid vector (tfidf weighted or not weighted)
        b) Doc2Vec
        c) PCA
        d) Autoencoder (e.g. deep autoencoder)
    2) Topic modeling:
        a) LDA
        b) LSI
    """
       
    def __init__(self, initial_documents):
        self.initial_documents = initial_documents
        self.corpus = []
        self.dictionary = []
        self.bow_corpus = []
        self.stopwords = []

        self.X_data = None
        
        # Trained models
        self.model_word2vec = None
        self.model_doc2vec = None
        self.model_lda = None
        self.model_lsi = None
        self.index_lda = None
        self.index_lsi = None
        self.tfidf = None
        self.vectors_centroid = []
        self.vectors_ae = []
        self.vectors_pca = []
        self.vectors_glove = None
        
        # Listed similarities
        self.list_similars_ctr = None
        self.list_similars_ctr_idx = None
        self.list_similars_ae = None
        self.list_similars_ae_idx = None
        self.list_similars_pca = None
        self.list_similars_pca_idx = None
        self.list_similars_lda = None
        self.list_similars_lda_idx = None
        self.list_similars_lsi = None
        self.list_similars_lsi_idx = None
        self.list_similars_d2v = None
        self.list_similars_d2v_idx = None
        self.list_similars_glove = None
        self.list_similars_glove_idx = None


    def preprocess_documents(self, max_fraction, min_frequency, 
                             remove_stopwords = None, create_stopwords = False):
        """ Preprocess 'documents'
        
        Obvious steps: 
            --> in 'helper_functions.preprocess_document'
            - Take all words that occur at least (min_frequency =) 2 times.
            - Lower case
            
        Calculate word frequency
        --> Words that occur more than max_fraction will become stopwords (words with no or little discriminative power)
        
        Args:
        --------
        max_fraction: float
            Gives maximum fraction of documents that may contain a certain word.
        min_frequency: int
            Words that occur less frequently will be ignored.
        remove_stopwords: list, None
            Give list of stopwords if they should be removed. Default is None.
        create_stopwords: bool
            if True: Words that are more common then max_fraction will be added to stopwords.
        """
        
        if max_fraction <= 0 or max_fraction > 1:
            print("max_fraction should be value > 0 and <= 1.")
        
        # Preprocess documents (all lower letters, every word exists at least 2 times)
        print("Preprocess documents...")
        if remove_stopwords is None:
            self.corpus, frequency = functions.preprocess_document(self.initial_documents, 
                                                                       stopwords = [], min_frequency = min_frequency)
        else:
            self.corpus, frequency = functions.preprocess_document(self.initial_documents, 
                                                                       stopwords = remove_stopwords, min_frequency = min_frequency)    
        
        # Create dictionary (or "vocabulary") containting all unique words from documents
        self.dictionary = corpora.Dictionary(self.corpus)
         
        if create_stopwords:
            # Calculate word frequency to determine stopwords
            print("Calculate inverse document frequency for entire dictionary.")
            documents_size = len(self.corpus)
            self.idf_scores = functions.ifd_scores(self.dictionary, self.corpus)
            
            # Words that appear too frequently (fraction>max_fration) become stopwords
            self.stopwords = self.idf_scores["word"][self.idf_scores["word count"] > documents_size*max_fraction]
            
            print(len(self.stopwords), " stopwords were selected from a total of ", 
                  len(self.dictionary), " words in the entire corpus.")
            
            # Create corpus, dictionary, and BOW corpus
            self.corpus, frequency = functions.preprocess_document(self.corpus, self.stopwords, min_frequency = min_frequency)
            
        self.bow_corpus = [self.dictionary.doc2bow(text) for text in self.corpus]



##
## -------------------- Model building & training  ----------------------------
## 
    def build_model_word2vec(self, file_model_word2vec, size=100, 
                             window=50, min_count=1, workers=4, 
                             iter=100, use_stored_model=True):
        """ Build Word2Vec model (using gensim)
        
        Args:
        --------
        file_model_word2vec: str,
            Filename to save model (or load model if it exists under this name).
        size: int,
            Dimensions of word vectors (default = 100) 
        window: int,
            Window size for context words (small for local context, 
            larger for global context, default = 50)
        min_count: int,
            Only consider words that occur at least min_count times in the corpus (default =1).
        workers: int,
            Number of threads to run the training on (should not be more than number of cores/threads, default = 4).
        iter: int,
            Number of training iterations (default=100). 
        use_stored_model: bool,
            Load stored model if True, else train new model.
        """
        
        epoch_logger = EpochLogger(iter)

        # Check if model already exists and should be loaded
        if os.path.isfile(file_model_word2vec) and use_stored_model:   
            print("Load stored word2vec model ...")
            self.model_word2vec = gensim.models.Word2Vec.load(file_model_word2vec)
        else:
            if use_stored_model:
                print("Stored word2vec model not found!")
            
            print("Calculating new word2vec model...")
        
            # Set up GENSIM logging
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
            # Train word2vec model
            self.model_word2vec = gensim.models.Word2Vec(self.corpus, size=size,
                                                         window=window, min_count=min_count, 
                                                         workers=workers, iter=iter,
                                                         seed=42, callbacks=[epoch_logger])
            
            # Save model
            self.model_word2vec.save(file_model_word2vec)


    def build_model_doc2vec(self, file_model_doc2vec, vector_size=100, window=50, 
                             min_count=1, workers=4, epochs=100, use_stored_model=True):
        """ Build Doc2Vec model (using gensim)
        
        Args:
        --------
        file_model_doc2vec: str,
            Filename to save model (or load model if it exists under this name).
        vector_size: int,
            Dimensions of word vectors (default = 100) 
        window: int,
            Window size for context words (small for local context, 
            larger for global context, default = 50)
        min_count: int,
            Only consider words that occur at least min_count times in the corpus (default =1).
        workers: int,
            Number of threads to run the training on (should not be more than number of cores/threads, default = 4).
        epochs: int,
            Number of training iterations (default=100). 
        use_stored_model: bool,
            Load stored model if True, else train new model.
        """
        from gensim.models.doc2vec import TaggedDocument
        
        epoch_logger = EpochLogger(epochs)
        
        # Check if model already exists and should be loaded
        if os.path.isfile(file_model_doc2vec) and use_stored_model:   
            print("Load stored doc2vec model ...")
            self.model_doc2vec = gensim.models.Doc2Vec.load(file_model_doc2vec)
        else:
            if use_stored_model:
                print("Stored doc2vec model not found!")
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.corpus)]
            print("Calculating new doc2vec model...")
            
            # Set up GENSIM logging
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
            
            # dm = 0 is for bag-of-words 
            self.model_doc2vec = gensim.models.Doc2Vec(documents, dm=0, vector_size=vector_size, 
                                    window=window, min_count=min_count, 
                                    workers=workers, epochs=epochs,
                                    seed=42, callbacks=[epoch_logger])
            
            # Save model
            self.model_doc2vec.save(file_model_doc2vec)
            
            
            
    def build_model_lda(self, file_model_lda, num_of_topics=100, num_pass=4, 
                        num_iter=100, use_stored_model=True):
        """ Build LDA model (using gensim)
        """
        
        # Check if model already exists and should be loaded
        if os.path.isfile(file_model_lda) and use_stored_model: 
            print("Load stored LDA model ...")
            self.model_lda = gensim.models.LdaModel.load(file_model_lda)
        else:
            if use_stored_model:
                print("Stored LDA model not found!")
            print("Calculating new LDA model...")
            self.model_lda = gensim.models.LdaModel(self.bow_corpus, id2word=self.dictionary, 
                                               num_topics=num_of_topics, passes=num_pass, iterations=num_iter) 
            
            # Save model
            self.model_lda.save(file_model_lda)
            
            # Output the Keyword in the 10 topics
            pprint("Keyword in the 10 topics")
            pprint(self.model_lda.print_topics())
        
        
    def build_model_lsi(self, file_model_lsi, num_of_topics=100, 
                        num_iter=10, use_stored_model=True):
        """ Build LDA model (using gensim)
        """
        
        # Check if model already exists and should be loaded
        if os.path.isfile(file_model_lsi) and use_stored_model: 
            print("Load stored LSI model ...")
            self.model_lsi = gensim.models.LsiModel.load(file_model_lsi)
        else:
            if use_stored_model:
                print("Stored LSI model not found!")
            print("Calculating new LSI model...")
            self.model_lsi = gensim.models.LsiModel(self.bow_corpus, 
                                                    id2word=self.dictionary, 
                                                    power_iters=num_iter,
                                                    num_topics=num_of_topics) 
            
            # Save model
            self.model_lsi.save(file_model_lsi)


    def build_autoencoder(self, file_model_ae, file_model_encoder, epochs = 100, batch_size = 1024, encoding_dim = 100,
                          layer_factors = (8,4,2)):
        """ Build and train a deep autoencoder model to reduce dimensionatliy of 
        corpus data.
        
        Args:
        --------
        file_model_ae: str
            Filename to save (or load if already present) autoencoder model.
        epochs: int
            Number of epochs to train autoencoder.
        batch_size: int
            Batch size for model training.
        encoding_dim: int
            Number of dimensions of the reduced representation.  
        layer_factors: (int, int, int)
            Layer dimensions of dense networks (factor with respect to encoding_dim).
        """
        # Check if model already exists and should be loaded
        if os.path.isfile(file_model_ae):   
            print("Load stored autoencoder ...")
            self.autoencoder = keras.models.load_model(file_model_ae)
            
            if os.path.isfile(file_model_encoder):
                train_new = False
                self.encoder = keras.models.load_model(file_model_encoder)
            else:
                train_new = True
        else:
            if len(file_model_ae) > 0:
                print("No stored model found!")
            train_new = True
            
        if train_new:
            print("Creating new autoencoder model...")
            
            input_dim = len(self.dictionary)
            layer1_factor, layer2_factor, layer3_factor = layer_factors
            
            corpus_dim = len(self.corpus)
            
            self.autoencoder = Sequential()
            
            # Encoder Layers
            self.autoencoder.add(Dense(layer1_factor * encoding_dim, input_shape=(input_dim,), activation='relu'))
            self.autoencoder.add(Dense(layer2_factor * encoding_dim, activation='relu'))
            self.autoencoder.add(Dense(layer3_factor * encoding_dim, activation='relu'))
            self.autoencoder.add(Dense(encoding_dim, activation='relu'))
            
            # Decoder Layers
            self.autoencoder.add(Dense(layer3_factor * encoding_dim, activation='relu'))
            self.autoencoder.add(Dense(layer2_factor * encoding_dim, activation='relu'))
            self.autoencoder.add(Dense(layer1_factor * encoding_dim, activation='relu'))
            self.autoencoder.add(Dense(input_dim, activation='sigmoid'))
            
            # Encoder
            input_img = Input(shape=(input_dim,))
            encoder_layer1 = self.autoencoder.layers[0]
            encoder_layer2 = self.autoencoder.layers[1]
            encoder_layer3 = self.autoencoder.layers[2]
            encoder_layer4 = self.autoencoder.layers[3]
            self.encoder = Model(input_img, encoder_layer4(encoder_layer3(encoder_layer2(encoder_layer1(input_img)))))
            
            # Display model architectures
            self.autoencoder.summary()
            self.encoder.summary()  
            
            self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
            # See if there is one-hot encoded vectors (X_data)
            if self.X_data is None:
                # Transform data to be used as input for Keras model
                self.X_data = np.zeros((corpus_dim, input_dim))
                
                for i, bow_doc in enumerate(self.bow_corpus[:corpus_dim]):
                    word_vector_bow = np.array([x[0] for x in bow_doc]).astype(int)
                    word_vector_count = np.array([x[1] for x in bow_doc]).astype(int)
                #    X_data[i,:] = one_hot_wv(dict_dim, word_vector_bow)
                    self.X_data[i,:] = functions.full_wv(input_dim, word_vector_bow, word_vector_count)
                
            # Shuffling
            shuffled_idx = np.arange(0, len(self.corpus))
            np.random.shuffle(shuffled_idx) 
            
            split_training = [0.8, 0.1, 0.1]
            print("split data into traning/testing/validation with fractions: ", split_training)
            
            split = [int(corpus_dim *split_training[0]), int(corpus_dim *(split_training[1]+split_training[0]))]
            
            x_train = self.X_data[shuffled_idx[:split[0]]]
            x_test = self.X_data[shuffled_idx[split[0]:split[1]]]
#            x_valid = self.X_data[shuffled_idx[split[1]:]]
    
            # Train autoencoder
            self.autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, x_test))
            
            # Save trained models
            self.autoencoder.save(file_model_ae) 
            self.encoder.save(file_model_encoder)
            
            
    def build_model_glove(self, file_model_glove, vector_size=300, window=100, 
                          learning_rate=0.05, workers=4, epochs=100, use_stored_model=True):
        """ Build glove model
        
        Args:
        --------
        file_model_glove: str,
            Filename to save model (or load model if it exists under this name).
        vector_size: int,
            Dimensions of word vectors (default = 100) 
        window: int,
            Window size for context words (small for local context, 
            larger for global context, default = 50)
        learning_rate: float,
            Learning rate for training of GloVe model (defaut set to 0.05).
        workers: int,
            Number of threads to run the training on (should not be more than number of cores/threads, default = 4).
        epochs: int,
            Number of training iterations (default=100). 
        use_stored_model: bool,
            Load stored model if True, else train new model.
        """
        from glove import Corpus, Glove
        
        # Check if model already exists and should be loaded
        if os.path.isfile(file_model_glove) and use_stored_model:   
            print("Load stored GloVe model ...")
            self.model_glove = Glove.load(file_model_glove)
        else:
            if use_stored_model:
                print("Stored GloVe model not found!")
            
            print("Calculating new GloVe model...")
        
            # Creating a corpus object
            corpus = Corpus() 
            
            # Training the corpus to generate the co occurence matrix which is used in GloVe
            MS_docs = self.corpus
            corpus.fit(MS_docs, window=window)

            self.model_glove = Glove(no_components=vector_size, learning_rate=learning_rate)          
            self.model_glove.fit(corpus.matrix, epochs=epochs, no_threads=workers, verbose=True)
            self.model_glove.add_dictionary(corpus.dictionary)
            
            # Save model
            self.model_glove.save(file_model_glove)

 
##
## -------------------- Calculate document vectors -------------------------------------
## 
    def get_vectors_centroid(self, extra_weights = None, tfidf_weighted=True):
        """ Calculate centroid vectors for all documents
        
        Individual word vectors are weighted using tfidf (unless weighted=False).
        
        Args:
        --------
        extra_weights: list
            List of extra weights for add documents (and every word). Set to "None" if not used.
        tfidf_weighted: bool
            True, False
        """
        
        #TODO: include extra weights!
        
        # Check if everything is there:
        if self.model_word2vec is None:
            print("Word2vec model first needs to be load or made (self.build_model_word2vec).")
        if len(self.bow_corpus) == 0:
            print("BOW corpus has not been calculated yet (bow_corpus).")
        
        self.tfidf = models.TfidfModel(self.bow_corpus)
        vector_size = self.model_word2vec.wv.vector_size
        vectors_centroid = []
        
        for i in range(len(self.bow_corpus)):
            if (i+1) % 10 == 0 or i == len(self.bow_corpus)-1:  # show progress
                print('\r', ' Calculated centroid vectors for ', i+1, ' of ', len(self.bow_corpus), ' documents.', end="")
            
            document = [self.dictionary[x[0]] for x in self.bow_corpus[i]]
            if extra_weights is not None:
                document_weight = [extra_weights[i][self.initial_documents[i].index(self.dictionary[x[0]])] for x in self.bow_corpus[i]]
                document_weight = np.array(document_weight)
                if len(document_weight) == 0:
                    print("Something might have gone wrong with: ", i)
                    np.ones((len(document)))
                else:
                    document_weight = np.sqrt(document_weight/np.max(document_weight))  # idea: take sqrt to make huge intensity differences less severe
            else:
                document_weight = np.ones((len(document)))
            if len(document) > 0:
                term1 = self.model_word2vec.wv[document]
                if tfidf_weighted:
                    term2 = np.array(list(zip(*self.tfidf[self.bow_corpus[i]]))[1])
                else:
                    term2 = np.ones((len(document)))
                    
                term1 = term1 * np.tile(document_weight, (vector_size,1)).T
                weighted_docvector = np.sum((term1.T * term2).T, axis=0)
            else:
                weighted_docvector = np.zeros((self.model_word2vec.vector_size))
            vectors_centroid.append(weighted_docvector)
            
        self.vectors_centroid = np.array(vectors_centroid)
         
#        # TODO add save and load options

        
    def get_vectors_pca(self, dimension=100):
        """ Calculate PCA vectors for all documents.
        
        Args:
        -------
        dimension: int
            Dimension of reduced PCA vectors. Default is 100. 
        """
        pca = PCA(n_components=dimension)
        
        input_dim = len(self.dictionary)
        corpus_dim = len(self.corpus)
        
        # See if there is one-hot encoded vectors (X_data)
        if self.X_data is None:
            # Transform data to be used as input for Keras model
            self.X_data = np.zeros((corpus_dim, input_dim))
            
            for i, bow_doc in enumerate(self.bow_corpus[:corpus_dim]):
                word_vector_bow = np.array([x[0] for x in bow_doc]).astype(int)
                word_vector_count = np.array([x[1] for x in bow_doc]).astype(int)
                self.X_data[i,:] = functions.full_wv(input_dim, word_vector_bow, word_vector_count)

        self.vectors_pca = pca.fit_transform(self.X_data)
        
        
    def get_vectors_glove(self, extra_weights = None, tfidf_weighted=True):
        """ Calculate GloVe vectors for all documents
        
        Individual word vectors are weighted using tfidf (unless weighted=False).
        
        Args:
        --------
        extra_weights: list
            List of extra weights for add documents (and every word). Set to "None" if not used.
        tfidf_weighted: bool
            True, False
        """
        #corpus = MS_measure.corpus  # TODO: use either corpus or bow_corpus, not both
        vector_size = self.model_glove.no_components
        vectors_glove = []
        
        for i in range(len(self.corpus)):
            if (i+1) % 10 == 0 or i == len(self.corpus)-1:  # show progress
                print('\r', ' Calculated GloVe vectors for ', i+1, ' of ', len(self.corpus), ' documents.', end="")
            
            document = [self.dictionary[x[0]] for x in self.bow_corpus[i]]
            if extra_weights is not None:
                document_weight = [extra_weights[i][self.initial_documents[i].index(self.dictionary[x[0]])] for x in self.bow_corpus[i]]
                document_weight = np.array(document_weight)
                if len(document_weight) == 0:
                    print("Something might have gone wrong with: ", i)
                    np.ones((len(document)))
                else:
                    document_weight = np.sqrt(document_weight/np.max(document_weight))  # idea: take sqrt to make huge intensity differences less severe
            else:
                document_weight = np.ones((len(document)))
            if len(document) > 0:
                term1 = np.zeros((len(document), vector_size))
                for m, doc in enumerate(document):
                    term1[m,:] = self.model_glove.word_vectors[self.model_glove.dictionary[doc]]
                if tfidf_weighted:
                    term2 = np.array(list(zip(*self.tfidf[self.bow_corpus[i]]))[1])
                else:
                    term2 = np.ones((len(document)))
                    
                term1 = term1 * np.tile(document_weight, (vector_size,1)).T
                weighted_docvector = np.sum((term1.T * term2).T, axis=0)
            else:
                weighted_docvector = np.zeros((vector_size))
            vectors_glove.append(weighted_docvector)
            
        self.vectors_glove = np.array(vectors_glove)
        

##
## -------------------- Calculate similarities -----------------------------------
## 
    def get_centroid_similarity(self, num_hits=25, method='cosine'):
        """ Calculate centroid similarities(all-versus-all --> matrix)
        
        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.      
        method: str
            See scipy spatial.distance.cdist for options. Default is 'cosine'.
        
        """
        list_similars_idx, list_similars, mean_similarity = functions.calculate_similarities(self.vectors_centroid, 
                                                                   num_hits, method = method)
        print("Calculated distances between ", list_similars.shape[0], " documents.")
        self.list_similars_ctr_idx = list_similars_idx
        self.list_similars_ctr = list_similars
        #self.mean_similarity_ctr = mean_similarity



    def get_autoencoder_similarity(self, num_hits=25, method='cosine'):
        """ Calculate autoencoder similarities(all-versus-all --> matrix)
        
        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.      
        method: str
            See scipy spatial.distance.cdist for options. Default is 'cosine'.
        
        """
        self.vectors_ae = self.encoder.predict(self.X_data)
        
        list_similars_ae_idx, list_similars_ae, mean_similarity = functions.calculate_similarities(self.vectors_ae, 
                                                                   num_hits, method = method)
        
        self.list_similars_ae_idx = list_similars_ae_idx
        self.list_similars_ae = list_similars_ae
        #self.mean_similarity_ae = mean_similarity


    def get_pca_similarity(self, num_hits=25, method='cosine'):
        """ Calculate PCA similarities(all-versus-all --> matrix)
        
        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.      
        method: str
            See scipy spatial.distance.cdist for options. Default is 'cosine'.
        
        """
        list_similars_idx, list_similars, mean_similarity = functions.calculate_similarities(self.vectors_pca, 
                                                                   num_hits, method = method)
        
        self.list_similars_pca_idx = list_similars_idx
        self.list_similars_pca = list_similars
        #self.mean_similarity_pca = mean_similarity
        
        
    def get_lda_similarity(self, num_hits=25):
        """ Calculate LDA topic based similarities (all-versus-all)
        
        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.      
        
        """

        # Now using faster gensim way (also not requiering to load everything into memory at once)
        index_tmpfile = get_tmpfile("index")
        index = gensim.similarities.Similarity(index_tmpfile, self.model_lda[self.bow_corpus], 
                                               num_features=len(self.dictionary))  # build the index
        Cdist = np.zeros((len(self.corpus), len(self.corpus)))
        for i, similarities in enumerate(index):  # yield similarities of all indexed documents
            Cdist[:,i] = similarities
            
#        Cdist = 1 - Cdist  # switch from similarity to distance

        # Create numpy arrays to store similarities
        list_similars_idx = np.zeros((Cdist.shape[0],num_hits), dtype=int)
        list_similars = np.zeros((Cdist.shape[0],num_hits))
        
        for i in range(Cdist.shape[0]):
            list_similars_idx[i,:] = Cdist[i,:].argsort()[-num_hits:][::-1]
            list_similars[i,:] = Cdist[i, list_similars_idx[i,:]]

        self.list_similars_lda_idx = list_similars_idx
        self.list_similars_lda = list_similars
        
        
    def get_lsi_similarity(self, num_hits=25):
        """ Calculate LSI based similarities (all-versus-all)
        
        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.      
        
        """

        # Now using faster gensim way (also not requiering to load everything into memory at once)
        index_tmpfile = get_tmpfile("index")
        index = gensim.similarities.Similarity(index_tmpfile, self.model_lsi[self.bow_corpus], 
                                               num_features=len(self.dictionary))  # build the index
        Cdist = np.zeros((len(self.corpus), len(self.corpus)))
        for i, similarities in enumerate(index):  # yield similarities of all indexed documents
            Cdist[:,i] = similarities
            
#        Cdist = 1 - Cdist  # switch from similarity to distance

        # Create numpy arrays to store distances
        list_similars_idx = np.zeros((Cdist.shape[0],num_hits), dtype=int)
        list_similars = np.zeros((Cdist.shape[0],num_hits))
        
        for i in range(Cdist.shape[0]):
            list_similars_idx[i,:] = Cdist[i,:].argsort()[-num_hits:][::-1]
            list_similars[i,:] = Cdist[i, list_similars_idx[i,:]]

        self.list_similars_lsi_idx = list_similars_idx
        self.list_similars_lsi = list_similars
        


    def get_doc2vec_similarity(self, num_hits=25, method='cosine'):
        """ Calculate Doc2Vec based similarities (all-vs-all)
        
        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.      
        method: str
            See scipy spatial.distance.cdist for options. Default is 'cosine'.
        """
        
        if self.vectors_glove is None:
            print("No trained Doc2Vec model found.")
            print("Please first train model using 'build_model_doc2vec' function.")
        else:
            vectors = np.zeros((len(self.corpus), self.model_doc2vec.vector_size))
            
            for i in range(len(self.corpus)):
                vectors[i,:] = self.model_doc2vec.docvecs[i]
    
            list_similars_idx, list_similars, mean_similarity = functions.calculate_similarities(vectors, 
                                                                       num_hits, method = method)
            
            self.list_similars_ctr_idx = list_similars_idx
            self.list_similars_ctr = list_similars
            #self.mean_similarity_ctr = mean_similarity


    def get_glove_similarity(self, num_hits=25, method='cosine'):
        """ Calculate GloVe based similarities (all-vs-all)
        
        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.      
        method: str
            See scipy spatial.distance.cdist for options. Default is 'cosine'.
        """
        
        if self.model_glove is None:
            print("No GloVe document vectors found.")
            print("Please first train model using 'build_model_glove' function.")
            print("Then create document vectors using 'get_vectors_glove' function.")
        else:
            list_similars_idx, list_similars, mean_similarity = functions.calculate_similarities(self.vectors_glove, 
                                                                       num_hits, method = method)
            print("Calculated distances between ", list_similars.shape[0], " documents.")
            
            self.list_similars_glove_idx = list_similars_idx
            self.list_similars_glove = list_similars


    def similarity_search(self, num_centroid_hits=100, centroid_min=0.3, 
                          doc2vec_refiner = True,
                          WMD_refiner = True):
        """ Search nearest neighbors in two steps:
        1- Use cosine measure based on LDA model to select candidates
        2- Calculate distances using alternative measures, such as
        document centroid cosine distance
        doc2vec similarity
        word movers distance (WMD) --> slow!
        """

        if num_centroid_hits > len(self.corpus):
            num_centroid_hits = len(self.corpus)
            print('number of best lda hits to keep is bigger than number of documents.')

        if self.model_lda is None:
            print("No lda model found. Calculate new one...")
        else:
            print("Lda model found.")

        if self.index_lda is None:
            print("No gensim LDA index found. Calculating new one...")
            self.index_lda = gensim.similarities.MatrixSimilarity(self.model_lda[self.bow_corpus])
        else:
            print("Gensim index found.") 
        
        if self.index_lsi is None:
            print("No gensim LSI index found. Calculating new one...")
            self.index_lsi = gensim.similarities.MatrixSimilarity(self.model_lsi[self.bow_corpus])
        else:
            print("Gensim LSI index found.") 
        

        # Calulate similarities based on LDA topics...
        candidates = []
#        keeptrack = []

        # TODO: Parallelize the following part of code!!
        # TODO: try smarter way of finding num_lda_hits based on random testing first and then a cutoff?
        for i, query in enumerate(self.corpus[:100]): # TODO: remove limit!
            # calculate distances between documents
            print('\r', 'Document ', i, ' of ', len(self.corpus), end="")

            # Even if slower, first take the most reliable similariy measure that's still OK to run (centroid)
            centroid_distances = np.zeros((len(self.corpus),2))
            centroid_distances[:, 0] = np.arange(0, len(self.corpus))
            for k in range(len(self.corpus)):
                centroid_distances[k, 1] = spatial.distance.cosine(self.vectors_centroid[i], self.vectors_centroid[k]) 
            
            centroid_distances = centroid_distances[np.lexsort((centroid_distances[:,0], centroid_distances[:,1])),:]
            
            # check that at least all close (<centroid_min) values are included 
            if np.where(centroid_distances[:,1] < centroid_min)[0].shape[0] > num_centroid_hits: 
                updated_hits = np.where(centroid_distances[:,1] < centroid_min)[0].shape[0]
            else: 
                updated_hits = num_centroid_hits
            
            candidate_idx = centroid_distances[:updated_hits, 0].astype(int) 
            
            distances = np.zeros((updated_hits, 6))
            distances[:,0] = np.array(candidate_idx) 
            distances[:,1] = centroid_distances[:updated_hits, 1]  #Centroid distances 
            
            # Calulate distances based on LDA topics...
            vec_bow = self.dictionary.doc2bow(query)
            vec_lda = self.model_lda[vec_bow]
            list_similars = self.index_lda[vec_lda]
            list_similars = sorted(enumerate(list_similars), key=lambda item: -item[1])
            distances[:,2] = np.array([x[1] for x in list_similars])[candidate_idx]
            
            # Calulate distances based on LSI...
            vec_lsi = self.model_lsi[vec_bow]
            list_similars_lsi = self.index_lsi[vec_lsi]
            list_similars_lsi = sorted(enumerate(list_similars_lsi), key=lambda item: -item[1])
            distances[:,3] = np.array([x[1] for x in list_similars_lsi])[candidate_idx]

  
            if WMD_refiner:
                for k, m in enumerate(candidate_idx):
                    distances[k,4] = self.model_word2vec.wv.wmdistance(self.corpus[i], self.corpus[m])

            if doc2vec_refiner:
                for k, m in enumerate(candidate_idx):
                    distances[k,5] = self.model_doc2vec.docvecs.similarity(i, m)

            distances[distances[:,1] > 1E10,2] = 1000 # remove infinity values
            distances = distances[np.lexsort((distances[:,3], distances[:,2], distances[:,1])),:]  # sort by centroid distance

            candidates.append(distances)
            # TODO switch to dataframe?
        return candidates

