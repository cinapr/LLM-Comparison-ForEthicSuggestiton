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

def clean_text(text):
    """Basic text cleaning to standardize the dataset."""
    text = text.lower()  # Convert text to lowercase
    text = ' '.join(text.split())  # Replace multiple spaces with a single space
    return text

def data_preparation(csv_data, prompt_template, response_template, output_jsonnl='data_for_finetuning.jsonl', columns_to_be_cleaned=None, style='chat-formatted', actAs='You are an ethics advisor focusing on ethical considerations in technology and AI.'):
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
            with open(output_jsonnl, 'w', encoding=fileencoding) as f:  # Ensure output is in UTF-8
                for index, row in df.iterrows():
                    #messages = [
                    #    {"role": "system", "content": actAs},
                    #    {"role": "user", "content": row["Questions"]},
                    #    {"role": "assistant", "content": row["ECCOLA_Code"]}
                    #]
                    prompt = prompt_template.format(**row)
                    response = response_template.format(**row)
                    messages = [
                        {"role": "system", "content": actAs},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    f.write(json.dumps({"messages": messages}) + '\n')
        else:
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
        #response = openai.FineTune.create(
        #    training_file=file_id,
        #    model=base_model,
        #    n_epochs=n_epochs,
        #    batch_size=batch_size,
        #    learning_rate_multiplier=learning_rate_multiplier
        #)
        response = client.fine_tuning.jobs.create(
            training_file=file_id, 
            model=base_model
        )

        # Assuming response provides a method or attribute to get the job_id
        if hasattr(response, 'fine_tuned_model'):
            #fine_tuned_job_id = response.fine_tuned_model
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
    
def retrieve_model_details(model_id):
    try:
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



def wait_for_fine_tuning_completion(job_id, timeout=3600, interval=60):
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
    



def use_model(model_id, user_story, max_tokens=150):
    try:
        prompt = f"Given this user story: '{user_story}' What ECCOLA type is most applicable, and what specific questions does it address?"
        #response = openai.Completion.create(
        response = client.chat.completions.create(
            model=model_id,
            prompt=prompt,
            max_tokens=max_tokens
        )
        #print(completion.choices[0].message)
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



def main2():
    ft_success1 = False
    #con ftjob-VUM6L21ykj4iHdoFkW6T0cEy
    #con2 ftjob-FiKDmSH3Z9yuz3e4gzDnyPvJ
    job_id1 = 'ftjob-FiKDmSH3Z9yuz3e4gzDnyPvJ' #'ftjob-LEKXFPEwoVs6KWpVUcaSknOO' #'ftjob-mn9iJLGpgvtwiYzVqoIprdyp' #None
    model_id1 = None
    model_id1_complete = False
    
    ft_success2 = False
    job_id2 = None
    model_id2 = None
    model_id1_complete = False

    # First Dataset
    if ((job_id1 is None) and (model_id1 is None)): #If questions have been trained, can skip this part
        prep_success1, jsonnl_file1 = data_preparation(csv_data='ECCOLA_Questions.csv', prompt_template="What does ECCOLA code {ECCOLA_Code} mean?", response_template="{Questions}", columns_to_be_cleaned=['ECCOLA_Code','Questions'])
        if prep_success1:
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
        if ((job_id2 is None) and (model_id2 is None)): #Skip if model2 tuned before
            # Sequentially fine-tune on the second dataset using the model fine-tuned on the first dataset
            prep_success2, jsonnl_file2 = data_preparation(csv_data='TrainingSet.csv', prompt_template="Classify this user story: {User_Story} {Acceptance_Criteria}", response_template="{ECCOLA_Code}", columns_to_be_cleaned=['ECCOLA_Code','User_Story','Acceptance_Criteria'])
            if prep_success2:
                upload_success2, upload_response2, file_id2 = upload_dataset(jsonnl_file2)
                if upload_success2:
                    ft_success2, ft_response2, job_id2 = fine_tune(file_id2, base_model=model_id1)
                    if ft_success2:
                        print("Model2 is currently being fine-tuned under job_id : ", job_id2)
                    else:
                        print("Failed in second phase of fine-tuning:", ft_response2)
                else:
                    print("Failed to upload second dataset:", upload_response2)
            else:
                print("Failed to prepare second dataset", jsonnl_file2)


        #Waiting for model2 finished to be fine-tuned
        model_id2_complete, model_id2 = wait_for_fine_tuning_completion(job_id2, timeout=3600, interval=30)


        if(model_id2_complete): #Only run when the first+second model finished to be tuned
            #storemodel
            storemodel_success, storemodel_response = store_model_id(model_id2)
            if(storemodel_success):
                print("Sequential fine-tuning completed successfully, model saved in : ", storemodel_response)
            else:
                print("Failed to save model :", storemodel_response)

            #accuracy check
            accuracy_success, accuracy_response2 = evaluate_model(model_id2, 'TestingDataSet.csv')
            if (accuracy_success):
                print("Accuracy of the model : ", accuracy_response2)
            else:
                print("Failed to check accuracy :", accuracy_response2)

            #Test classification
            # Example of using the model
            user_story1 = "As a security officer who got contacted by the police about a certain car, I want to track where the car was seen last, so I can predict where it is going next."
            use_success, result = use_model(model_id2, user_story1)
            if use_success:
                print("Classification and questions:", result)
            else:
                print("Failed to classify story:", result)
    

def main():
    retrieve_model_details('ft:gpt-3.5-turbo-0125:personal::9RMWiWgi')

if __name__ == '__main__':
    main()
