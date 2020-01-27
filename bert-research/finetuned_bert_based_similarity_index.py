import torch
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
import os
import time, datetime

def check_gpu_set_device():
    try:
        if torch.cuda.is_available:
          print ('Number of GPUs is :',torch.cuda.device_count())
          print ('Name of GPU is :', torch.cuda.get_device_name())
          device = torch.device("cuda")
        else:
          device = torch.device("cpu")
    except : 
        print ('Exception occured while checking for GPU support..')
        device = torch.device("cpu")
    
    return device

def load_questions_data():
    data = pd.read_csv('qna_log.csv')
    data.head()

    questions = data.Question.values
    categories = data.Category.values
    
    return questions, categories

# TODO: Instead of single question, modify implementation to pass in BATCHES
def generate_embedding(model, tokenizer, question, device):

    # ===========================
    #    STEP 1: Preparing Input Data for BERT
    # ===========================

    # Encode each question
    #
    #   1. Add special tokens for [CLS], [SEP]
    #   2. Set max length of question as 64
    #   3. Pad each question to max length

    input_id = tokenizer.encode(question,
                            add_special_tokens=True,
                            max_length=64,
                            pad_to_max_length=True)


    # create attention mask list for each question
    attention_mask = []

    # iterate over each token in the tokenized question
    for t in input_id:
        if t>0:
            attention_mask.append(1)
        else:
            attention_mask.append(0)

    # create a torch tensor representation for tokenized question
    input_id = torch.tensor(input_id).unsqueeze(0)

    # create a torch tensor representation for attention masks
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)

    assert len(input_id) == len(attention_mask)

    # ===========================
    #    STEP 2: BERT Model
    # ===========================

    #set model for evaluation mode
    model.eval()

    # Telling the model not to build the backwards graph will make this a little quicker.
    with torch.no_grad():

        # load input_id to the device (cpu or gpu)
        input_id = input_id.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass, return hidden states and predictions.
        # This will return the logits rather than the loss because we have
        # not provided labels.
        logits, encoded_layers = model (input_id,
                                        token_type_ids=None,
                                        attention_mask=attention_mask)
        
        layer = 11 # The last BERT layer before classifier
        batch = 0 # The first sentence, there is just one input to this batch
        token = 0 # The first token, corresponds to [CLS]

        # extract embedding of [CLS] token from last layer
        embedding = encoded_layers[layer][batch][token]

        # detach embedding from GPU and load to CPU
        embedding = embedding.cpu().numpy()

        return embedding

def load_saved_model_tokenizer(output_dir):
    
    ####### Load saved model from disk #######    

    model = BertForSequenceClassification.from_pretrained(
        output_dir,
        output_hidden_states = True, # Whether the model returns all hidden-states
    )

    tokenizer = BertTokenizer.from_pretrained(output_dir)
    try:
        model.cuda()
    except:
        print ('Torch not compiled with cuda, ignore')

    return model, tokenizer

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def save_embeddings(embeddings):

    output_dir = './output_dir/bert/qna-embeddings/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use numpy to write out the matrix of embeddings.
    print("Saving embeddings to: {:}".format(output_dir))
    np.save(output_dir+'embeddings.npy', embeddings)

def main():
    # get connected device
    device = check_gpu_set_device()
    
    # TODO: This should come as argument
    output_dir = './output_dir/bert/'
    
    # load already fine-tuned model and tokenizer
    model, tokenizer = load_saved_model_tokenizer(output_dir)
    
    # load questions data
    questions, categories = load_questions_data()

    embeddings = []
    t0 = time.time()

    for i in range (len(questions)):

        if i%50==0 :
            print ('\tGenerating embeddings {:}/{:}'.format(i+1, len(questions)))

        embedding = generate_embedding(model, tokenizer, questions[i], device)
        embeddings.append(embedding)
    
    save_embeddings(embeddings)
    print ('Generated {:} embeddings in {:}'.format(len(questions), format_time(time.time()-t0)))    

#if __name__ == "__main__":
main()