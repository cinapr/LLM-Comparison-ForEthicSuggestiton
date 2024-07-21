import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2Config, GPT2ForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import os
import time



class TextClassificationDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',  # Ensure that all samples are padded to the same length
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Adjust dimensions if necessary
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }




# Define load_data function
def clean_text(text):
    """ Basic text cleaning """
    # Implement actual cleaning based on your dataset specifics
    text = text.lower().strip()
    text = ' '.join(text.split())  # Remove unnecessary spaces
    return text

def augment_text(text):
    """ Placeholder for text augmentation logic """
    # Implement paraphrasing or back-translation if possible
    return text

def preprocess_data(df):
    df['User_Story'] = df['User_Story'].apply(clean_text)
    df['Acceptance_Criteria'] = df['Acceptance_Criteria'].apply(clean_text)
    df['combined_text'] = df['User_Story'] + ' ' + df['Acceptance_Criteria']
    df['combined_text'] = df['combined_text'].apply(augment_text)
    return df



def load_data(file_path):
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    texts = df['combined_text'].tolist()
    categories = df['ECCOLA_Code'].tolist()
    unique_labels = set(categories)
    category_to_id = {category: idx for idx, category in enumerate(unique_labels)}
    labels = [category_to_id[category] for category in categories]
    return texts, labels, len(unique_labels)

def save_model(model, tokenizer, model_path):
    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), f"{model_path}/model_state.bin")
    
    # Save model using the save_pretrained method instead of torch.save for compatibility with from_pretrained
    model.save_pretrained(model_path)

    # Save tokenizer and its configuration
    tokenizer.save_pretrained(model_path)
    
    # Save configuration of the model
    model.config.save_pretrained(model_path)

def evaluate_accuracy(trainer, dataset):
    results = trainer.evaluate(eval_dataset=dataset)
    print("Evaluation Results:", results)
    return results

'''
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(model.device)
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

def evaluate_accuracy(trainer, dataset, model, training_accuracy_check=False):
    if training_accuracy_check:
        eval_results = trainer.evaluate()
        print("Trainer Accuracy:", eval_results)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")
'''

def main():
    train_file_path = "D:\\SE4GD\\Semester4\\Code\\GPT2\\TrainingSet.csv"
    eval_file_path = "D:\\SE4GD\\Semester4\\Code\\GPT2\\TestingDataSet.csv"
    
    start_time = time.time()

    train_texts, train_labels, num_labels = load_data(train_file_path)
    eval_texts, eval_labels, _ = load_data(eval_file_path)

    #tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=512)

    # instantiate the configuration for your model, this can be imported from transformers
    configuration = GPT2Config()

    # Model must be loaded or created after setting the tokenizer's pad_token
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
    #model = GPT2ForSequenceClassification(configuration).from_pretrained('gpt2', num_labels=num_labels)
    #model = GPT2ForSequenceClassification(configuration).from_pretrained(model_name).to(device)
    
    # Ensure the model's embeddings are resized
    model.resize_token_embeddings(len(tokenizer))
    #model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = model.config.eos_token_id # set the pad token of the model's configuration

    # Create datasets using the properly configured tokenizer
    train_dataset = TextClassificationDataset(tokenizer, train_texts, train_labels)
    eval_dataset = TextClassificationDataset(tokenizer, eval_texts, eval_labels)

    # Setup DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    #model = BartForSequenceClassification.from_pretrained('facebook/bart-large', num_labels=num_labels)
    
    # Training arguments setup
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Start training
    trainer.train()

    # Save model and tokenizer
    model_path = 'D:\\SE4GD\\Semester4\\Code\\GPT2\\gpt2_finetuned'
    save_model(model, tokenizer, model_path)
    
    # Optionally, save the dataset using pickle for analysis
    with open('D:\\SE4GD\\Semester4\\Code\\GPT2\\dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

    end_time = time.time()
    print(f"GPT2 Training Time: {end_time - start_time:.4f} seconds")

    # Evaluate the model
    evaluate_accuracy(trainer, eval_dataset)


    

if __name__ == '__main__':
    main()


'''
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['User_Story'] = df['User_Story'].apply(clean_text)
    df['Acceptance_Criteria'] = df['Acceptance_Criteria'].apply(clean_text)
    # Simple augmentation by duplicating hard-to-classify samples
    hard_classes = df[df['ECCOLA_Code'].isin(['hard_to_classify_codes'])]
    df_augmented = pd.concat([df, hard_classes])
    return df_augmented

# Using a smaller model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

# Adjust training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,  # Increased epochs with early stopping
    per_device_train_batch_size=8,  # Smaller batch size
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="steps",
    eval_steps=50,  # Evaluate the model every 50 steps
    save_strategy="no",  # Save only the best model at the end
    load_best_model_at_end=True,
    metric_for_best_model='f1',  # Use F1 score to determine the best model
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics  # Define this function to calculate F1, precision, and recall
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted'),
    }

'''