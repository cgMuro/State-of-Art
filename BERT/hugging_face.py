# https://huggingface.co/transformers/training.html#pytorch

import torch
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Get GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------- FINE-TUNING ----------------------------------- #
# Get BERT model instance with encoder weights copied from the "bert-base-uncased" model and a randomly initialized sequence 
# classification head on top of the encoder with an output size of 2.
model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
# Put model in training mode
model.train()

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
# Apply different parameters to a specific parameter group: we apply weight decay to all parameters except bias and layer normalization terms
no_decay = ['bias', 'LayerNorm.weight']
optimizer_groundeed_parameters = [
    { 'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01 },
    { 'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0 }
]

# Define tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Define dummy data batch for training
text_batch = ["I love Pixar.", "I don't care for Pixar."]
# Tokenize the text_batch
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
# Get the indexes of the tokenizer
input_ids = encoding['input_ids']
# Get the attention mask
attention_mask = encoding['attention-mask']

# Create dummy labels
labels = torch.tensor([1, 0]).unsqueeze(0)

# Define learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,  # Pass in the optimizer
    num_warmup_steps=num_warmup_steps,  # Number of steps for the warmup phase, after which the learning rate decreases linearly
    num_training_steps=num_training_steps  # Total number of training steps
)

# Get prediction from model
outputs = model(
    input_ids,  # Pass the data batch encoded
    attention_mask=attention_mask, # Pass the attention mask
    labels=labels  # Pass the labels for the classification  task
)
# Get loss
loss = outputs.loss  # or equivalent -> loss = F.cross_entropy(outputs.logits, labels)
# Calculate gradients
loss.backward()
# Update weights
optimizer.step()
# Update the scheduler
scheduler.step()


# ----------------------------------- FREEZING THE ENCODER ----------------------------------- #
# To keep the weights of the pre-trained encoder frozen and optimizing only the weights of the head layers.
for param in model.base_model.parameters():
    param.requires_grad = False


# ----------------------------------- TRAINER ----------------------------------- #
# Simple but feature-complete training and evaluation interface. 
# You can train, fine-tune, and evaluate any 🤗 Transformers model with a wide range of training options and with built-in features.

# Get pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-large-uncased')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',           # Output directory
    num_train_epochs=3,               # Total number of training epochs
    per_device_train_batch_size=16,   # Batch size per device during training
    per_device_eval_batch_size=64,    # Batch size for evaluation
    warmup_steps=500,                 # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                # Strength of weight deecay
    logging_dir='./logs'              # Directory for storing logs
)
# Define trainer
trainer = Trainer(
    model=model,                      # The instantiated 🤗 Transformers model to be trained
    args=training_args,               # Training arguments
    train_dataset=train_dataset,      # Training dataset
    eval_dataset=test_dataset         # Evaluation dataset
)

# Start training
trainer.train()

# Start evaluating
trainer.evaluate()

# Define additional metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
