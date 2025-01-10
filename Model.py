#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install transformers datasets torch


# In[7]:


pip install --upgrade transformers


# In[1]:


import transformers
print(transformers.__version__)  # Should be at least 4.30 or later


# In[5]:


pip install transformers


# In[6]:


pip install torch


# In[3]:


import pandas as pd

# Load the dataset

# Display the first few rows
print(data.head())
data['prompt'] = data['Environmental Factor'].apply(lambda x: f"predict: {x}")
data['output'] = data['Yield (kg)']  # Replace with your target column


# In[4]:


from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(data['prompt'], data['output'], test_size=0.2)


# In[6]:


pip install sentencepiece


# In[1]:


import sentencepiece
print(sentencepiece.__version__)  # This should print the installed version


# In[15]:


pip install transformers[torch]


# In[4]:


# Generate prompt and output columns
import pandas as pd
data = pd.read_csv('synthetic_apple_data.csv')
data['prompt'] = data.apply(lambda row: f"predict: Temperature {row['Temperature (C)']}C, Rainfall {row['Rainfall (mm)']}mm, Soil Moisture {row['Soil Moisture (%)']}%", axis=1)
data['output'] = data['Yield (kg)'].astype(str)  # Ensure output is a string

# Split data for training
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(data['prompt'], data['output'], test_size=0.2)

# Tokenize the data
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)
import torch

class AppleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = tokenizer(list(labels), truncation=True, padding=True).input_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx]),
        }

train_dataset = AppleDataset(train_encodings, train_labels)
val_dataset = AppleDataset(val_encodings, val_labels)
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments

model = T5ForConditionalGeneration.from_pretrained('t5-small')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()


# In[ ]:




