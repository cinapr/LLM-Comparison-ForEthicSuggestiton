from transformers import BartTokenizer, BartForSequenceClassification, pipeline
import pandas as pd
import os
import time

ECCOLA_CODE_NAMES = {
    0: "#0 Stakeholder Analysis",
    1: "#1 Types of Transparency",
    2: "#2 Explainability",
    3: "#3 Communication",
    4: "#4 Documenting Trade-offs",
    5: "#5 Traceability",
    6: "#6 System Reliability",
    7: "#7 Privacy and Data",
    8: "#8 Data Quality",
    9: "#9 Access to Data",
    10: "#10 Human Agency",
    11: "#11 Human Oversight",
    12: "#12 System Security",
    13: "#13 System Safety",
    14: "#14 Accessibility",
    15: "#15 Stakeholder Participation",
    16: "#16 Environmental Impact",
    17: "#17 Societal Effects",
    18: "#18 Auditability",
    19: "#19 Ability to Redress"
}

def normalize_label(label):
    # Remove spaces and convert to lower case
    return label.replace(" ", "").lower()


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified model path does not exist: {model_path}")
    print("Files in the directory:", os.listdir(model_path))
    model = BartForSequenceClassification.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, framework='pt')
    return classifier

def classify_text(text, classifier):
    result = classifier(text)
    for prediction in result:
        label_index = int(prediction['label'].split('_')[-1])
        prediction['label'] = ECCOLA_CODE_NAMES[label_index]
    return result

def main():
    model_path = "D:\\SE4GD\\Semester4\\Code\\BART\\bart_finetuned\\"
    classifier = load_model(model_path)

    from_file = True
    if from_file:
        data_path = "D:\\SE4GD\\Semester4\\Code\\BART\\RealDataSet.csv"
        df = pd.read_csv(data_path)
        correct_count = 0
        total_count = 0

        for index, row in df.iterrows():
            start_time = time.time()

            input_text = row['User_Story'] + ' ' + row['Acceptance_Criteria']
            predicted_result = classify_text(input_text, classifier)
            predicted_label = normalize_label(predicted_result[0]['label'])
            actual_label = normalize_label(row['ECCOLA_Code'])

            end_time = time.time()

            is_correct = predicted_label == actual_label
            print(f"Row {index}: Prediction: {predicted_label}, Actual: {actual_label}, Correct: {is_correct}, Time: {end_time - start_time:.4f} seconds")
            
            if is_correct:
                correct_count += 1
            total_count += 1

        print(f"Total Correct: {correct_count}")
        print(f"Total Incorrect: {total_count - correct_count}")
        print(f"Accuracy: {correct_count / total_count:.2%}")
    
    else:
        input_text = ""
        while(input_text != "QUIT"):
            start_time = time.time()
            input_text = input("Give me the user stories? ")
            predictions = classify_text(input_text, classifier)
            print(f"Predictions: {predictions}")
            end_time = time.time()
            print(f"Time taken for answering: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    main()

