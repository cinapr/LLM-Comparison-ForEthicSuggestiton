import openai
import pandas as pd
import os
import time

#API_KEY = 'your-api-key-here'
API_KEY = 'your-api-key-here'

openai.api_key = API_KEY

# Mapping of numerical labels to descriptive labels
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
    # Normalize the label by removing spaces and converting to lower case
    return label.replace(" ", "").lower()

def load_model_id(file_path="model_id.txt"):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error loading model ID: {e}")
        return None

def use_model(model_id, user_story, max_tokens=150):
    try:
        prompt = f"Given this user story: '{user_story}' What ECCOLA type is most applicable, and what specific questions does it address?"
        response = openai.Completion.create(model=model_id, prompt=prompt, max_tokens=max_tokens)
        return True, response.choices[0].text.strip()
    except Exception as e:
        return False, str(e)

def main():
    model_id = load_model_id()
    if not model_id:
        print("Model ID could not be loaded.")
        return
    
    from_file = True
    if from_file:
        data_path = "D:\\SE4GD\\Semester4\\Code\\GPT2\\RealDataSet.csv"
        df = pd.read_csv(data_path)
        correct_count = 0
        total_count = 0

        for index, row in df.iterrows():
            start_time = time.time()
            input_text = row['User_Story'] + ' ' + row['Acceptance_Criteria']

            success, result = use_model(model_id, input_text)
            if success:
                print("Classification and questions:", result)
                predicted_result = result
                predicted_label = normalize_label(predicted_result[0]['label'])
                actual_label = normalize_label(row['ECCOLA_Code'])

                end_time = time.time()

                is_correct = predicted_label == actual_label
                print(f"Row {index}: Prediction: {predicted_label}, Actual: {actual_label}, Correct: {is_correct}, Time: {end_time - start_time:.4f} seconds")
                if is_correct:
                    correct_count += 1
            else:
                print("Failed to classify story:", result)

            total_count += 1

            
            

        print(f"Total Correct: {correct_count}")
        print(f"Total Incorrect: {total_count - correct_count}")
        print(f"Accuracy: {correct_count / total_count:.2%}")
    
    else:
        input_text = ""
        while(input_text != "QUIT"):
            start_time = time.time()

            user_story = input("Give me the user stories? ")
            success, result = use_model(model_id, user_story)
            if success:
                print("Classification and questions:", result)
            else:
                print("Failed to classify story:", result)

            end_time = time.time()
            print(f"Time taken for answering: {end_time - start_time:.4f} seconds")

    

if __name__ == '__main__':
    main()
