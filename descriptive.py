#%%
from functions import load_datasets

#%%
train_data, val_data, test_data = load_datasets()

train_text = train_data["tweet"] 
val_text = val_data["tweet"]
test_text = test_data["tweet"]

train_labels = train_data["label"]
val_labels = val_data["label"]
test_labels = test_data["label"]


#%%
label_counts = train_data['label'].value_counts()
print(label_counts)

number_samples = train_data.shape[0]

print(label_counts[0] / number_samples)
print(label_counts[1] / number_samples)
print(label_counts[2] / number_samples)


#%%





