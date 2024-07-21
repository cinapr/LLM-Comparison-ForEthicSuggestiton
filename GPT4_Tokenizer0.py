import openai
import pandas as pd
import json
import datetime

# Define your OpenAI API key at the top of the script
API_KEY = 'your-openai-api-key-here'

# Initialize the OpenAI API key
openai.api_key = API_KEY

def clean_text(text):
    """Basic text cleaning to standardize the dataset."""
    text = text.lower()  # Convert text to lowercase
    text = ' '.join(text.split())  # Replace multiple spaces with a single space
    return text

def data_preparation(csv_data, prompt_template, response_template, output_jsonnl='data_for_finetuning.jsonl', columns_to_be_cleaned=None):
    fileencoding = 'utf-8'
    try:
        # Attempt to read the CSV file with a different encoding if UTF-8 fails
        try:
            df = pd.read_csv(csv_data, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_data, encoding='utf-8-sig')
                fileencoding = 'utf-8-sig'
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(csv_data, encoding='cp1252')  # Common in Windows-generated files
                    fileencoding = 'cp1252'
                except Exception as e1:
                    return False, str(e1)

        # Clean specified columns
        if columns_to_be_cleaned:
            for column in columns_to_be_cleaned:
                if column in df.columns:
                    df[column] = df[column].apply(clean_text)
                    
        with open(output_jsonnl, 'w', encoding=fileencoding) as f:  # Ensure output is in UTF-8
            for index, row in df.iterrows():
                prompt = prompt_template.format(**row)
                response = response_template.format(**row)
                f.write(json.dumps({"prompt": prompt, "completion": response}) + '\n')
        return True, output_jsonnl
    except Exception as e:
        return False, str(e)


def upload_dataset(file_path):
    try:
        response = openai.File.create(file=file_path, purpose='fine-tune')
        return True, response, response['id']
    except Exception as e:
        return False, str(e), None




def fine_tune(file_id, base_model="text-davinci-002", n_epochs=4, batch_size=4, learning_rate_multiplier=0.1):
    try:
        response = openai.FineTune.create(
            training_file=file_id,
            model=base_model,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier
        )
        return True, response, response['fine_tuned_model']
    except Exception as e:
        return False, str(e), None




def store_model_id(model_id, file_path_prefix="model_id"):
    try:
        file_path = f"{file_path_prefix}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        with open(file_path, 'w') as file:
            file.write(model_id)
        return True, file_path
    except Exception as e:
        return False, str(e)
    



def use_model(model_id, user_story, max_tokens=150):
    try:
        prompt = f"Given this user story: '{user_story}' What ECCOLA type is most applicable, and what specific questions does it address?"
        response = openai.Completion.create(
            model=model_id,
            prompt=prompt,
            max_tokens=max_tokens
        )
        return True, response.choices[0].text.strip()
    except Exception as e:
        return False, str(e)



def evaluate_model(model_id, test_data_csv):
    try:
        df = pd.read_csv(test_data_csv)
        correct_predictions = 0
        total_predictions = len(df)

        for index, row in df.iterrows():
            user_story = row['User_Story']
            actual_code = row['ECCOLA_Code']
            success, prediction = use_model(model_id, user_story)
            if success and "ECCOLA Type: " + actual_code in prediction:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        return True, str(accuracy)

    except Exception as e:
        return False, str(e)

def main():
    # First Dataset
    prep_success1, jsonnl_file1 = data_preparation(csv_data='ECCOLA_Questions.csv', prompt_template="What does ECCOLA code {ECCOLA_Code} mean?", response_template="{Questions}", columns_to_be_cleaned=['ECCOLA_Code','Questions'])
    if prep_success1:
        upload_success1, upload_response1, file_id1 = upload_dataset(jsonnl_file1)
        if upload_success1:
            ft_success1, ft_response1, model_id1 = fine_tune(file_id1)
            if ft_success1:
                # Sequentially fine-tune on the second dataset using the model fine-tuned on the first dataset
                prep_success2, jsonnl_file2 = data_preparation(csv_data='TrainingSet.csv', prompt_template="Classify this user story: {User_Story} {Acceptance_Criteria}", response_template="{ECCOLA_Code}", columns_to_be_cleaned=['ECCOLA_Code','User_Story','Acceptance_Criteria'])
                if prep_success2:
                    upload_success2, upload_response2, file_id2 = upload_dataset(jsonnl_file2)
                    if upload_success2:
                        ft_success2, ft_response2, model_id2 = fine_tune(file_id2, base_model=model_id1)
                        if ft_success2:
                            accuracy_success, accuracy_response2 = evaluate_model(model_id2, 'TestingDataSet.csv')
                            print("Accuracy of the model : ", accuracy_response2)

                            storemodel_success, storemodel_response = store_model_id(model_id2)
                            print("Sequential fine-tuning completed successfully.")
                        
                            # Example of using the model
                            user_story1 = "As a security officer who got contacted by the police about a certain car, I want to track where the car was seen last, so I can predict where it is going next."
                            use_success, result = use_model(model_id2, user_story1)
                            if use_success:
                                print("Classification and questions:", result)
                            else:
                                print("Failed to classify story:", result)

                        else:
                            print("Failed in second phase of fine-tuning:", ft_response2)
                    else:
                        print("Failed to upload second dataset:", upload_response2)
                else:
                    print("Failed to prepare second dataset", jsonnl_file2)
            else:
                print("Failed to fine-tune first dataset:", ft_response1)
        else:
            print("Failed to upload first dataset:", upload_response1)
    else:
        print("Failed to prepare first dataset", jsonnl_file1)

if __name__ == '__main__':
    main()
