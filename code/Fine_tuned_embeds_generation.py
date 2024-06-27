#pip install torch

import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import warnings
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

warnings.filterwarnings('ignore')
csv.field_size_limit(10000000) 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer = AutoTokenizer.from_pretrained('./Fine_Tuned_Transformer_1_10')
# model = AutoModelForSequenceClassification.from_pretrained('./Fine_Tuned_Transformer_1_10').to(device)


def generate_embeds(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)

    # Get the last hidden states
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

    sentence_embedding = last_hidden_states.mean(dim=1).cpu()
    
    return sentence_embedding 

def sentence_embeds(text):
    sentence_ls = sent_tokenize(text)
    
    embeds_ls = []
    
    for text in sentence_ls:
        
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(device)

        # Get the last hidden states
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]

        sentence_embedding = last_hidden_states.mean(dim=1).cpu()
        embeds_ls.append(sentence_embedding.numpy()[0])
    
    return embeds_ls




################# sentence level ####################
# with open('matched_df.csv') as f:
    
#     reader = csv.reader(f)
    
#     #skip the header
#     next(reader)
    
#     embed_res = []
    
#     progress_bar = tqdm(total=57257, desc="Encoding messages")

#     for row in reader:
        
#         message = row[4]
        
#         embed_res.append(sentence_embeds(message))
                         
#         progress_bar.update(1)
                         
#     progress_bar.close()
                         
#     df = pd.DataFrame()
#     df['embeddings'] = embed_res
    
#     df.to_parquet('sentence_fine_tuned_qa_case_embeds.parquet')







############## query level ####################
##### QA ######
# with open('full_fine_tuned_qa_case_embeds.npy', 'wb') as res, open('case_clean.csv') as f:
    
#     reader = csv.reader(f)
    
#     #skip the header
#     next(reader)
    
#     embed_res = []
    
#     progress_bar = tqdm(total=1699508, desc="Encoding messages")

#     for row in reader:
        
#         message = row[4]
        
#         embed_res.append(generate_embeds(message).numpy())
                         
#         progress_bar.update(1)
                         
#     progress_bar.close()
                         
#     np.save(res, embed_res)





# Define chunk size
# chunk_size = 200000

# # Open the CSV file
# with open('case_clean.csv') as f:
#     reader = csv.reader(f)
    
#     # Skip the header
#     next(reader)
    
#     # Initialize variables
#     embed_res = []
#     chunk_number = 1
#     progress_bar = tqdm(total=1699508, desc="Encoding messages")
    
#     # Read and process the file in chunks
#     for i, row in enumerate(reader):
        
#         message = row[4]
        
#         # Add generated embeddings to the list
#         embed_res.append(generate_embeds(message).numpy())
#         progress_bar.update(1)
        
#         # If the current chunk is complete, save it
#         if (i + 1) % chunk_size == 0:
#             np.save(f'fine_tuned_qa_case_embeds_updated_chunk_{chunk_number}.npy', embed_res)
#             embed_res = []  # Reset the list for the next chunk
#             chunk_number += 1

#     # Save any remaining embeddings that didn't make a full chunk
#     if embed_res:
#         np.save(f'fine_tuned_qa_case_embeds_updated_chunk_{chunk_number}.npy', embed_res)
    
#     progress_bar.close()

##### QQ ######
from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-mpnet-base-v2')

with open('full_qq_case_embeds.npy', 'wb') as res, open('case_clean.csv') as f:
    
    reader = csv.reader(f)
    
    #skip the header
    next(reader)
    
    embed_res = []
    
    progress_bar = tqdm(total=1699508, desc="Encoding messages")

    for row in reader:
        
        message = row[4]
        
        embed_res.append(model.encode(message))
                         
        progress_bar.update(1)
                         
    progress_bar.close()
                         
    np.save(res, embed_res)