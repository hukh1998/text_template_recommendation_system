import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import torch

case_new = pd.read_csv('matched_df.csv')
unmatched_df = pd.read_csv('unmatched_df.csv')

model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# generate matched and unmatched dataset 
inquiry = list(case_new.cleaned_description.values)
response = list(case_new.cleaned_message.values) 
targets = [1] * len(inquiry)

for i in range(unmatched_df.shape[0]):
    inquiry.append(unmatched_df.cleaned_description.values[i])
    response.append(unmatched_df.cleaned_MessageBody.values[i])
    targets.append(0)
    
test_df = pd.DataFrame({"question": inquiry, "response": response, "targets": targets})

sample_size = 1
case_sample = test_df.sample(frac=sample_size, random_state=42)
question = list(case_sample.question.values)
answer = list(case_sample.response.values)
labels = list(case_sample.targets.values)



# model training 

from torch.utils.data import DataLoader, TensorDataset

# Tokenize
inputs = tokenizer(question, answer, return_tensors="pt", padding=True, truncation=True, max_length=128)

# to tensor
labels = torch.tensor(labels).long()  

# Create DataLoader
dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm 

NUM_EPOCHS = 10
epoch_loss_ls = []

# loss function and optimizer
loss_function = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# move to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# model training
model.train()
for epoch in tqdm(range(NUM_EPOCHS)):
    
    running_loss = 0
    
    print(f"Starting epoch {epoch + 1}")
    for i, batch in enumerate(dataloader):
        input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_function(outputs.logits, batch_labels)
        
        #print(f"  Batch {i + 1}, Loss: {loss.item()}")
        
        loss.backward()
 
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_loss = running_loss
    epoch_loss_ls.append(epoch_loss)
    
    print("Epoch: {}/{} \t Training Loss: {}".format(epoch+1,NUM_EPOCHS, epoch_loss))
    
# save the model
path = f'./Fine_Tuned_Transformer_{sample_size}_{NUM_EPOCHS}/'

# 'model' is an instance of MPNetForSequenceClassification or any other transformers model
model.save_pretrained(path)

# If you have a tokenizer associated with the model, you can save it too:
tokenizer.save_pretrained(path)
