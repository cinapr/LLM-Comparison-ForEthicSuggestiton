# train_model.py

import pandas as pd
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
import pickle  
import time

class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = (df['User_Story'] + ' ' + df['Acceptance_Criteria']).tolist()
    categories = df['ECCOLA_Code'].tolist()
    unique_labels = set(categories)

    print(texts)
    print(categories)
    print(unique_labels)

    category_to_id = {category: idx for idx, category in enumerate(unique_labels)}
    labels = [category_to_id[category] for category in categories]
    return texts, labels, len(unique_labels)
    





# 12. Evaluate the model
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
    accuracy = correct / total
    return accuracy





def evaluate_accuracy(trainer, dataset, model, training_accuracy_check=False):
    #OPEN ONLY WHEN RUNNING TRAINER AS WELL | CLOSE WHEN RUN ONLY ACCURACY CHECK
    if (training_accuracy_check):
        eval_results = trainer.evaluate()
        print("Trainer Accuracy: ")  # Access the accuracy from the dictionary
        print(eval_results)

    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")




#Train + Evaluate
def main():
    start_time = time.time()

    # Training data
    train_file_path = "D:\\SE4GD\\Semester4\\Code\\BART\\TrainingSet.csv"
    train_texts, train_labels, num_labels = load_data(train_file_path)

    # Evaluation data
    eval_file_path = "D:\\SE4GD\\Semester4\\Code\\BART\\TestingDataSet.csv"
    eval_texts, eval_labels, _ = load_data(eval_file_path)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # Tokenize training data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    train_dataset = TextClassificationDataset(train_encodings, train_labels)

    # Tokenize evaluation data
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=512)
    eval_dataset = TextClassificationDataset(eval_encodings, eval_labels)

    model = BartForSequenceClassification.from_pretrained('facebook/bart-large', num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset  # Provide evaluation dataset here
    )

    trainer.train()

    # Explicitly save model and tokenizer
    #model.save_pretrained('./bart_finetuned')
    model.save_pretrained('D:\\SE4GD\\Semester4\\Code\\BART\\bart_finetuned')
    tokenizer.save_pretrained('D:\\SE4GD\\Semester4\\Code\\BART\\bart_finetuned')

    with open('dataset.pkl', 'wb') as f:  # Save the dataset object using pickle
        pickle.dump(train_dataset, f)

    end_time = time.time()
    print(f"BART Training Time: {end_time - start_time:.4f} seconds")

    # Evaluate accuracy
    evaluate_accuracy(trainer, eval_dataset, model, True)  # Pass eval_dataset for evaluation



#Evaluate only
def main_evaluateOnly():
    # Load the pre-trained model
    model = BartForSequenceClassification.from_pretrained('./bart_finetuned')

    # Load the evaluation dataset
    eval_file_path = "D:\\SE4GD\\Semester4\\Code\\BART\\RealDataSet.csv"
    eval_texts, eval_labels, _ = load_data(eval_file_path)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=512)
    eval_dataset = TextClassificationDataset(eval_encodings, eval_labels)

    # Create a trainer object (not needed for evaluation, but depends on your setup)
    trainer = Trainer(
        model=model
    )

    # Evaluate accuracy
    evaluate_accuracy(trainer, eval_dataset, model, False)




if __name__ == '__main__':
    main()

