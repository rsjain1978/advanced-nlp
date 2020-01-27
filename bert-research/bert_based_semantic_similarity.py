import numpy as np

def load_embeddings():

    output_dir = './output_dir/bert/qna-embeddings/'
    
    print("Loading embeddings to: {:}".format(output_dir))

    # Use numpy to write out the matrix of embeddings.
    embeddings = np.load(output_dir+'embeddings.npy')

    print ('Loaded embeddings with shape {:}'.format(embeddings.shape))
    
load_embeddings()