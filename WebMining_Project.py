# %%
import pandas as pd
import numpy as np
import torch
#from tensorflow.keras.optimizers import Adam
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax
from torch.utils.data import DataLoader
from transformers import AdamW
from functions import load_datasets
from sklearn.metrics import f1_score

import evaluate
import optuna
from optuna.trial import Trial


#%% 
USE_GPU = True
print(torch.__version__)
print("checking if GPU is available ")
print(torch.cuda.is_available())

print("check number of GPUs available on machine and number of devices used ")
print(torch.cuda.device_count())
#print(torch.cuda.current_device())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} is available (CUDA device name).")
    print(torch.cuda.get_device_properties(0))
else:
    device = torch.device("cpu")
    print("GPU not available, training on CPU")


# %%
# train_text = pd.read_table('train_text.txt', header=None)
# train_text = train_text.rename(columns={0: "Tweets"})

# val_text = pd.read_table('val_text.txt', header=None)
# val_text = val_text.rename(columns={0: "Tweets"})

# test_text = pd.read_table('test_text.txt', header=None)
# test_text = test_text.rename(columns={0: "Tweets"})

# train_labels = pd.read_table('train_labels.txt', header=None)
# train_labels = train_labels.rename(columns={0: "Label"})

# val_labels = pd.read_table('val_labels.txt', header=None)
# val_labels = val_labels.rename(columns={0: "Label"})

# test_labels = pd.read_table('test_labels.txt', header=None)
# test_labels = test_labels.rename(columns={0: "Label"})

train_data, val_data, test_data = load_datasets()

train_text = train_data["tweet"].to_list() 
val_text = val_data["tweet"].to_list()
test_text = test_data["tweet"].to_list()

train_labels = train_data["label"].to_list()
val_labels = val_data["label"].to_list()
test_labels = test_data["label"].to_list()



print("train_text: " + str(len(train_text)) + ", train_labels: " + str(len(train_labels)) + ", val_text: " + str(len(val_text)) 
      + ", val_labels: " + str(len(val_labels)) + ", test_text: " + str(len(test_text)) + ", test_labels: " + str(len(test_labels)))



def tokenize_function(text):
  tokenized_list = []

  # extract text
  for t in text:
      tokenized_text = tokenizer(
          t,
          return_tensors='pt',
          truncation=True,
          padding=True,
          max_length=512
          )
      tokenized_list.append(tokenized_text)

  return tokenized_list


# %%
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# train_encodings = tokenize_function(train_text.to_string())
# val_encodings = tokenize_function(val_text.to_string())
# test_encodings = tokenize_function(test_text.to_string())


max_length = 280
train_encodings = tokenizer(train_text, padding=True, truncation=True, max_length=max_length)
val_encodings = tokenizer(val_text, padding=True, truncation=True, max_length=max_length)
test_encodings = tokenizer(test_text, padding=True, truncation=True, max_length=max_length)


print(train_encodings['input_ids'][2524])

# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        y_matrix = torch.tensor(self.labels[idx])
        if USE_GPU:
            # print("using gpu")
            #item = {key: x_train.cuda() for key, x_train in item.items()}
            #y_matrix = y_matrix.cuda()
            item = {key: x_train.to(device) for key, x_train in item.items()}
            y_matrix = y_matrix.to(device)
        item['labels'] = y_matrix
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)
test_dataset = Dataset(test_encodings, test_labels)
#######################################################################################################################
# single model part
#######################################################################################################################
# %%

# dataloader_pin_memory set to False due to GPU not working otherwise
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=30,
    #gradient_accumulation_steps=2,
    dataloader_pin_memory = False
)

# %%

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer_res = trainer.train()


# %%
# evaluate the finetuned model 
trainer.evaluate()

# %%
# making the predictions with the test dataset 
trainer_predictions = trainer.predict(test_dataset=test_dataset)

# %%
# getting the label predictions out of the trainer object/class 
preds = np.argmax(trainer_predictions.predictions, axis=-1)

# %%
# defining the output metrics for the test score 
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
accuracy_metric = evaluate.load("accuracy")

# computing the defined score metrics 
f1 = f1_metric.compute(predictions=preds, references=trainer_predictions.label_ids, average='weighted')
precision = precision_metric.compute(predictions=preds, references=trainer_predictions.label_ids, average='weighted')
recall = recall_metric.compute(predictions=preds, references=trainer_predictions.label_ids, average='weighted')
accuracy = accuracy_metric.compute(predictions=preds, references=trainer_predictions.label_ids)

# print the results of the score metrics 
print(f1, precision, recall, accuracy)

# %%
# Only used for pretrained model withour finetuning. 
def score_function(model, test_data, test_labels, device):
    i = 0
    label_pred = []
    model.eval()
    model.cpu()
    for t in test_data:
    # model_input = tokenize_function(t)
        output = model(**t)
        scores = output.logits.detach().numpy()
        scores = softmax(scores)
        print(i)
        print(scores)
        
        neg_score = scores[0][0]
        neu_score = scores[0][1]
        pos_score = scores[0][2]

        if neg_score > neu_score and neg_score > pos_score:
            label_pred.append(0)
        elif neu_score > neg_score and neu_score > pos_score:
            label_pred.append(1)
        elif pos_score > neu_score and pos_score > neg_score:
            label_pred.append(2)
        i = i+1

    f1 = f1_score(test_labels, label_pred, average='weighted')
    return f1

# %%
# Only used for the pretrained model without finetuning 
model_input = tokenize_function(test_text)
print(model_input)

# %%
# Only used for the pretrained model without finetuning 
score = score_function(model, model_input, test_labels, device)

# %%
# Only used for the pretrained model without finetuning 
print(score)


# %%
'''device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()'''

#######################################################################################################################
#  the hyperparameter search part
#######################################################################################################################
# %%

# learning rate  5e-5, 4e-5, 3e-5, and 2e-5 5e-4
# epochs 3 5

# Grid search 
def model_init():
    return model

def optuna_hp_space(trial: Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("per_device_train_batch_size", 2, 5),
        "evaluation_strategy": trial.suggest_categorical("evaluation_rate", ["steps", "epoch"])

    }

# def compute_objective(metrics):
#     return metrics["eval_loss"]


def compute_metrics(eval_preds):
        metric = evaluate.load("f1")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")
    

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=30,
    #gradient_accumulation_steps=2,
    dataloader_pin_memory = False,
    #load_best_model_at_end=True,
    metric_for_best_model="f1",
    evaluation_strategy="steps"
    
)

# %%

trainer = Trainer(
    model_init=model_init,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)



best_trials = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10
    #compute_objective=compute_objective,
)

print("Best trial: {}".format(best_trials.objective))

print("with parameters:")
for key, value in best_trials.hyperparameters.items():
    print("{}: {}".format(key, value))


# optuna dashboard command
#  optuna-dashboard sqlite:///db.sqlite3
# %%
