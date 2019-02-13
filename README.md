# SeSiMe
#### (Sentence/Sequence Similarity Measures)

Protoype name. And prototyoe code.

Here used to calculate similarities (or distances) between mass spectra or between biosynthetic gene clusters (BGCs).


## Method categories

| Method        | Sensitive to word context | Sensitive to word order |
| ------------- | ------------- |------------- |
| PCA on 1-hot document vector | No | No |
| Autoencoder on 1-hot document vector | No | No |
| Autoencoder on full sequence | (Yes) | (Yes) |
| Word2Vec + document centroid | Yes | No |
| ELMo, bi-LSTM etc. | Yes | Yes |

