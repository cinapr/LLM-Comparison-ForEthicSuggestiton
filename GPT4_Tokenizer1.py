import openai
from openai import OpenAI
import pandas as pd
import json
import datetime
import time

# Define your OpenAI API key at the top of the script
API_KEY = 'your-api-key-here'

# Initialize the OpenAI API key
#openai.api_key = API_KEY

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=API_KEY)

# Define the acceptable ECCOLA_Codes
ECCOLA_CODES_DICT = [
    "#0 Stakeholder Analysis",
    "#1 Types of Transparency",
    "#2 Explainability",
    "#3 Communication",
    "#4 Documenting Trade-offs",
    "#5 Traceability",
    "#6 System Reliability",
    "#7 Privacy and Data",
    "#8 Data Quality",
    "#9 Access to Data",
    "#10 Human Agency",
    "#11 Human Oversight",
    "#12 System Security",
    "#13 System Safety",
    "#14 Accessibility",
    "#15 Stakeholder Participation",
    "#16 Environmental Impact",
    "#17 Societal Effects",
    "#18 Auditability",
    "#19 Ability to Redress",
    "#20 Minimizing Negative Impacts"
]




#Cleaning for training - Have space for better view
def clean_text(text):
    """Basic text cleaning to standardize the dataset."""
    text = text.lower()  # Convert text to lowercase
    text = ' '.join(text.split())  # Replace multiple spaces with a single space
    return text

#Cleaning for evaluation of result, no space to prevent error due to space available
def clean_text_delete_midSpace(text):
    """Basic text cleaning to standardize the dataset."""
    text = text.lower()  # Convert text to lowercase
    text = ' '.join(text.split())  # Replace multiple spaces with a single space
    text = text.strip().replace(" ", "") 
    return text

def data_preparation(csv_data, prompt_template, response_template, reason_template=None, output_jsonnl='data_for_finetuning.jsonl', columns_to_be_cleaned=None, style='chat-formatted', filemethod='w', actAs='You are an ethics advisor focusing on ethical considerations in technology and AI.'):
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

        # Clean specified columns if necessary
        if columns_to_be_cleaned:
            for column in columns_to_be_cleaned:
                if column in df.columns:
                    df[column] = df[column].apply(clean_text)

        if(style=='chat-formatted'):
            with open(output_jsonnl, filemethod, encoding=fileencoding) as f:  # Ensure output is in UTF-8
                for index, row in df.iterrows():
                    #messages = [
                    #    {"role": "system", "content": actAs},
                    #    {"role": "user", "content": row["Questions"]},
                    #    {"role": "assistant", "content": row["ECCOLA_Code"]}
                    #]
                    prompt = prompt_template.format(**row)
                    response = response_template.format(**row)
                    if(reason_template is None):
                        messages = [
                            {"role": "system", "content": actAs},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}
                        ]
                    else:
                        reason = reason_template.format(**row)
                        messages = [
                            {"role": "system", "content": actAs},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                            {"role": "assistant", "content": reason}
                        ]
                    f.write(json.dumps({"messages": messages}) + '\n')
        else:
            with open(output_jsonnl, filemethod, encoding=fileencoding) as f:  # Ensure output is in UTF-8
                for index, row in df.iterrows():
                    prompt = prompt_template.format(**row)
                    response = response_template.format(**row)
                    if(reason_template is not None):
                        reason = reason_template.format(**row)
                        prompt = "Because of " + prompt + " is answering " + reason
                    f.write(json.dumps({"prompt": prompt, "completion": response}) + '\n')
        return True, output_jsonnl
    
    except Exception as e:
        return False, str(e)




def upload_dataset(file_path):
    try:
        response = client.files.create(
            file=open(file_path, "rb"),
            purpose="fine-tune"
        )
        file_id = response.id  # Accessing file ID
        #file_id = response['id']  # Ensure the file ID is retrieved correctly
        
        return True, response, file_id
    except Exception as e:
        return False, str(e), None




def fine_tune(file_id, base_model="gpt-3.5-turbo", n_epochs=4, batch_size=4, learning_rate_multiplier=0.1):
    #base_model = text-davinci-002, gpt-3.5-turbo, babbage-002, davinci-002
    try:
        response = client.fine_tuning.jobs.create(
            training_file=file_id, 
            model=base_model
        )

        # Assuming response provides a method or attribute to get the job_id
        if hasattr(response, 'fine_tuned_model'):
            fine_tuned_job_id = response.id
            fine_tuned_model_type = response.model
            fine_tuned_model_object = response.object
            fine_tuned_model_organization = response.organization_id
            fine_tuned_model_trainingfiles = response.training_file
            fine_tuned_model_trainedtoken = response.trained_tokens
            fine_tuned_model_status = response.status
        else:
            fine_tuned_job_id = None  # Or handle accordingly if no model ID is available

        return True, response, fine_tuned_job_id
        #return True, response, response['fine_tuned_model']
    except Exception as e:
        return False, str(e), None
    


#After you've started a fine-tuning job, it may take some time to complete. 
#Your job may be queued behind other jobs in our system, and training a model can take minutes or hours depending on the model and dataset size. 
#After the model training is completed, the user who created the fine-tuning job will receive an email confirmation.
#In addition to creating a fine-tuning job, you can also list existing jobs, retrieve the status of a job, or cancel a job.
def queue_model_finetuning_queue(queuelimit=10):
    finetuningjobs = client.fine_tuning.jobs.list(limit=queuelimit) # List 10 fine-tuning jobs
    return finetuningjobs

def queue_model_finetuning_details(job_id):
    finetuningstate = client.fine_tuning.jobs.retrieve(job_id) # Retrieve the state of a fine-tune
    return finetuningstate

def queue_model_finetuning_status(job_id):
    job_details = client.fine_tuning.jobs.retrieve(job_id)  # Assuming this function returns the job details including the status
    if job_details:
        status = job_details.status
        if status == "succeeded":
            model_id = job_details.fine_tuned_model
            print("The fine-tuning job (",job_id,") completed successfully : ", model_id)
            return 1, model_id
        elif status == "running":
            print("The fine-tuning job is currently running : ", job_id)
            return 2, job_id
        elif status == "failed":
            print("The fine-tuning job failed : ", job_id)
            print(f"Error Message: ", job_details.error)
            return 0, job_id
        elif status == "cancelled":
            print("The fine-tuning job was cancelled : ", job_id)
            return 0, job_id
        else:
            if((job_details.error.code is not None) and (job_details.error.code != '')):
                print("The fine-tuning job is stopped due to failure [", status ,"] : ", job_id)
                print(f"Error Message: ", job_details.error)
                return 0, job_id
            else:
                print("The fine-tuning job is in the queue [", status ,"] : ", job_id)
                return 2, job_id
    else:
        print("Failed to retrieve job details or job does not exist : ", job_id)
        return 0, job_id

def retrieve_fine_tuning_job(job_id):
    try:
        finetuning_jobdetails = client.fine_tuning.jobs.retrieve(job_id) # Retrieve the state of a fine-tune
        print("Fine-tuning job details:")
        print(f"Status: {finetuning_jobdetails['status']}")
        print(f"Model: {finetuning_jobdetails['model']}")
        print(f"Created at: {finetuning_jobdetails['created_at']}")
        print(f"Training loss: {finetuning_jobdetails['training']['final_loss'] if 'training' in finetuning_jobdetails and 'final_loss' in finetuning_jobdetails['training'] else 'N/A'}")
        return finetuning_jobdetails
    except Exception as e:
        print(f"An error occurred while retrieving the job: {str(e)}")
        return None


def retrieve_models(model_id=None):
    try:   
        if (model_id is None):
            # List fine-tuned models
            fine_tuned_models = client.models.list()
            print(fine_tuned_models)
            return fine_tuned_models
        else:
            model_details = client.models.retrieve(model_id)
            print("Model details retrieved successfully.")
            print(f"Model ID: {model_details.id}")
            print(f"Model Object: {model_details.object}")
            print(f"Model Created At: {model_details.created}")
            print(f"Model Owned By: {model_details.owned_by}")
            print(f"Model Fields: {model_details.model_computed_fields}")
            print(f"Model Config: {model_details.model_config}")
            print(f"Model Object: {model_details.object}")
            #print(f"Model Permission: {model_details.permission}")
            print(f"Model: {model_details}")
            return model_details
    except Exception as e:
        print(f"An error occurred while retrieving the model details: {str(e)}")
        return None
    

    
def queue_model_finetuning_cancelmodel(job_id):
    finetuningcancel = client.fine_tuning.jobs.cancel(job_id) # Cancel a job
    return finetuningcancel

def queue_model_finetuning_eventslist(job_id, queuelimit=10):
    finetuningevents = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, queuelimit=10) # List up to 10 events from a fine-tuning job
    return finetuningevents



def wait_for_fine_tuning_completion(job_id, timeout=3600, interval=10):
    """Wait for fine-tuning job to complete within a timeout period.
    
    Args:
        job_id (str): The fine-tuning job ID.
        timeout (int): Total time to wait before timeout (in seconds).
        interval (int): Time between status checks (in seconds).
    
    Returns:
        bool: True if job completed successfully, False if failed or timeout.
        dict: The final job details if completed, error message otherwise.
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            model_id = None
            model_id_complete,model_id = queue_model_finetuning_status(job_id)
            if(model_id_complete == 0):
                return False, job_id
            elif(model_id_complete == 1):
                print("Complete fine-tuning model (",model_id,") in ", str(time.time() - start_time), " seconds.")
                return True, model_id
        except Exception as e:
            print(f"An error occurred while retrieving the job: {str(e)}")
            return False, str(e)
        time.sleep(interval)
    
    print("Timeout reached while waiting for fine-tuning job ",job_id," to complete.")
    return False, {"error": "timeout"}



def store_model_id(model_id, file_path_prefix="model_id"):
    try:
        file_path = f"{file_path_prefix}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        with open(file_path, 'w') as file:
            file.write(model_id)
        return True, file_path
    except Exception as e:
        return False, str(e)
    


def use_model(model_id, user_story, max_tokens=150, prompt = None, actAs='You are an ethics advisor focusing on ethical considerations in technology and AI.', cleaned_eccola_codes=None):
    try:
        # Prepare the message for the completion request
        if (prompt == None):
            #prompt2 = f"Which ECCOLA Questions, that answered by this user story: '{user_story}'?"
            prompt1 = f"Which ECCOLA Questions be answered by this user story: '{user_story}', and what is the ECCOLA_Code that correlated to that questions/user stories?"
            messages = [
                {"role": "system", "content": actAs},
                #{"role": "user", "content": prompt2},
                {"role": "user", "content": prompt1},
                {"role": "user", "content": f"The acceptable ECCOLA_Codes are: {', '.join(ECCOLA_CODES_DICT)}. Please respond with the most applicable ECCOLA_Code in the format '#<number> <title>'."}
            ]
        else:
            messages = [
                {"role": "system", "content": actAs},
                {"role": "user", "content": prompt},
                {"role": "user", "content": f"The acceptable ECCOLA_Codes are: {', '.join(ECCOLA_CODES_DICT)}. Please respond with the most applicable ECCOLA_Code in the format '#<number> <title>'."}
            ]
        
        if cleaned_eccola_codes is None:
            cleaned_eccola_codes = [clean_text_delete_midSpace(code) for code in ECCOLA_CODES_DICT]

        for _ in range(5):  # Try up to 5 times
            # Call the OpenAI API for a completion
            #response = openai.Completion.create(
            #response = openai.ChatCompletion.create(
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens
            )
            
            #Get the prediction from GPT AI
            content = response.choices[0].message.content.strip()
            
            # Clean and check the content against the ECCOLA_CODES_DICT
            cleaned_content = clean_text_delete_midSpace(content.split("\n")[0])
            
            #If the result in the classification dict then return if not tried again until 5 times
            if cleaned_content in cleaned_eccola_codes:
                # Return the response
                return True, content
                #return True, response.choices[0].message['content'].strip()
                #return True, response.choices[0].message
                #return True, response.choices[0].message.content.strip()
                #return True, response.choices[0].text.strip()
        
        # If after 5 attempts no valid response was received
        return False, "Failed to get a valid ECCOLA_Code after 5 attempts."
    except Exception as e:
        return False, str(e)
    
        
        
        
    
    
    



def evaluate_model(model_id, test_data_csv):
    try:
        df = pd.read_csv(test_data_csv)
        correct_predictions = 0
        total_predictions = len(df)

        cleaned_eccola_codes = [clean_text_delete_midSpace(code) for code in ECCOLA_CODES_DICT]

        for index, row in df.iterrows():
            user_story = row['User_Story']
            actual_code = row['ECCOLA_Code']
            success, prediction = use_model(model_id=model_id, user_story=user_story, cleaned_eccola_codes=cleaned_eccola_codes)
            if success and clean_text_delete_midSpace(actual_code) == clean_text_delete_midSpace(prediction.split("\n")[0]):
                correct_predictions += 1
            else:
                print("["+actual_code+"], Predicted: ["+prediction+"], "+user_story)

        accuracy = correct_predictions / total_predictions
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        return True, str(accuracy) + "/" + str(total_predictions)

    except Exception as e:
        return False, str(e)



def main():
    ft_success1 = False
    #job_id1 = None
    job_id1 = None
    model_id1 = None
    model_id1_complete = False
    
    ft_success2 = False
    job_id2 = None
    model_id2 = None
    model_id1_complete = False

    start_time = time.time()
    # First Dataset
    if ((job_id1 is None) and (model_id1 is None)): #If questions have been trained, can skip this part
        prep_success11, jsonnl_file1 = data_preparation(csv_data='ECCOLA_Questions.csv', prompt_template="If the INPUT Paragraph can answer the: {Questions}", response_template="{ECCOLA_Code}", columns_to_be_cleaned=['ECCOLA_Code','Questions'], filemethod='w')
        prep_success12, jsonnl_file1 = data_preparation(csv_data='ECCOLA_QuestionsCorrelations.csv', prompt_template="If the INPUT Paragraph related to: {Questions_Reasoning}", response_template="{ECCOLA_Code}", columns_to_be_cleaned=['ECCOLA_Code','Questions_Reasoning'], filemethod='a')
        prep_success13, jsonnl_file1 = data_preparation(csv_data='TrainingSet.csv', prompt_template="INPUT is = {User_Story} {Acceptance_Criteria}. What previously defined classifications of ECCOLA_Code related to the INPUT?", response_template="{ECCOLA_Code}", reason_template="Because, the INPUT ({User_Story} {Acceptance_Criteria}) can answer question:{Related_Questions}, which are classified under {ECCOLA_Code}", columns_to_be_cleaned=['ECCOLA_Code','User_Story','Acceptance_Criteria', 'Related_Questions'], filemethod='a')

        if prep_success11 and prep_success12 and prep_success13:
            upload_success1, upload_response1, file_id1 = upload_dataset(jsonnl_file1)
            if upload_success1:
                ft_success1, ft_response1, job_id1 = fine_tune(file_id1)
                if ft_success1:
                    print("Model1 is currently being fine-tuned under job_id : ", job_id1)
                else:
                    print("Failed to fine-tune first dataset:", ft_response1)
            else:
                print("Failed to upload first dataset:", upload_response1)
        else:
            print("Failed to prepare first dataset", jsonnl_file1)

    #Waiting for model1 finished to be fine-tuned
    model_id1_complete, model_id1 = wait_for_fine_tuning_completion(job_id1, timeout=3600, interval=60)

    #Second dataset
    if(model_id1_complete == False): #Only run when the first model finished to be tuned
        print("Failed to complete first fine tuning")
    else:
        #storemodel
        storemodel_success, storemodel_response = store_model_id(model_id1)#model_id2)
        if(storemodel_success):
            print("Fine-tuning completed successfully, model saved in : ", storemodel_response)
            end_time = time.time()
            print(f"GPT4 Training Time: {end_time - start_time:.4f} seconds")
        else:
            print("Failed to save model :", storemodel_response)

        #accuracy check
        accuracy_success, accuracy_response2 = evaluate_model(model_id1, 'TestingDataSet.csv') #model_id2, 'TestingDataSet.csv')
        if (accuracy_success):
            print("Accuracy of the model : ", accuracy_response2)
        else:
            print("Failed to check accuracy :", accuracy_response2)

        #Test classification
        # Example of using the model
        user_story1 = "As a security officer who got contacted by the police about a certain car, I want to track where the car was seen last, so I can predict where it is going next."
        use_success, result = use_model(model_id1, user_story1) #model_id2, user_story1)
        if use_success:
            print("Classification and questions:", result)
        else:
            print("Failed to classify story:", result)
    

def test():
    trained_model_id = 'ft:gpt-3.5-turbo-0125:personal::9Rn8dKhf'
    retrieve_models(trained_model_id) #ft:gpt-3.5-turbo-0125:personal::9RMWiWgi')
    
    user_story = 'As a system user, I need guielines to understand the system functionalities. Provide guidelines document'
    _, response = use_model(model_id=trained_model_id, user_story=user_story, max_tokens=150)
    print(response)

    accuracy_success, accuracy_response2 = evaluate_model(trained_model_id, 'TestingDataSet.csv') #model_id2, 'TestingDataSet.csv')
    if (accuracy_success):
        print("Accuracy of the model : ", accuracy_response2)
    else:
        print("Failed to check accuracy :", accuracy_response2)


if __name__ == '__main__':
    main()
    #test()
