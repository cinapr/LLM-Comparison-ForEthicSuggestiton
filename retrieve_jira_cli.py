import requests
import json
import re  # Import the regular expression module
import csv

class retrieve_jira:
    def __init__(self):#, parent):
        #self.parent = parent
        pass

    #CONNECT TO JIRA API
    def connect(self, API_route, params=None):
        # JIRA settings
        jira_domain = "XXX.atlassian.net"
        api_token = "put_jira_api_token"
        email = "put_your_email"

        # API Endpoint
        url = f'https://{jira_domain}{API_route}' 
        print(url)

        auth = (email, api_token)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        # JQL Query
        if (params is not None):
            # Send the request
            response = requests.get(url, auth=auth, headers=headers, params=params)
        else:
            # Send the request
            response = requests.get(url, auth=auth, headers=headers)


        # Check for errors and print the project keys
        if response.status_code == 200:
            return response
        else:
            ErrorMessage = f"Failed to retrieve response: {response.status_code} - {response.text}"
            print(ErrorMessage)
            return None


    #RETRIEVE JIRA-ID
    def main_retrieve(self, project_key = "SCRUM", board_id = None, printIDs=False):
        try:
            # project_key = "SCRUM"
            # board_id = 1 Resilience
            jql_query = {
                'jql': f'project = {project_key}',
                'fields': 'id,key'
            }
            response = self.connect(API_route=f'/rest/api/3/search', params=jql_query)
            if response is not None:
                issues_dict = {}  # Initialize an empty dictionary to store issues
                issues = response.json().get('issues', [])
                issue_amount = 0
                for issue in reversed(issues):
                    issue_id = issue['id']
                    issue_key = issue['key']
                    issues_dict[issue_id] = issue_key
                    issue_amount += 1
                
                # Print the issues from the dictionary
                if(printIDs == True):
                    for issue_id, issue_key in issues_dict.items():
                        print(f"Issue ID: {issue_id}, Key: {issue_key}")

                return True, issues_dict, str(issue_amount)

        except Exception as e:
            failure_msg = f"Fail to retrieve JIRA IDs: {e}"
            print(failure_msg)
            return False, None, failure_msg
        


    #RETRIEVE DETAILS OF JIRA-ID
    def extract_comments(self, comments_json):
        comments_list = []
        
        # Traverse through each comment's body content
        for comment in comments_json:
            body = comment['body']
            for content_block in body['content']:  # Access the outer content block which is the ordered list or bullet list
                if content_block['type'] == 'orderedList' or content_block['type'] == 'bulletList':
                    for item in content_block['content']:  # Iterate through each list item in the content
                        paragraph_content = []
                        for paragraph in item['content']:  # Access content within each list item
                            for text_block in paragraph['content']:  # Each text block within a paragraph
                                if text_block['type'] == 'text':
                                    paragraph_content.append(text_block['text'])
                                elif text_block['type'] == 'hardBreak':
                                    paragraph_content.append('\n')  # Add a new line for hard breaks
                        # Join all parts of the list item into a single string and add it to the list
                        comments_list.append(' '.join(paragraph_content))
        
        return comments_list
    
    def details_retrieve(self, issue_key, params=None, getComments=False, getAttachments=False, printDetails=False):
        try:
            # fields = Fields to fetch: Adjust these fields based on the exact names in your JIRA setup if different
            #fields = 'id,assignee,created,status,resolutiondate,description,summary,comment,attachment'
            # params = List of API Fields to be retrieved taken from Retrieve_APIFields_JIRA 
            # params={'fields': fields}
            
            # JIRA settings
            # issue_key = 'SCRUM-55'  # Use SCRUM-8 or the issue ID if preferred

            # Send the request
            response = self.connect(API_route=f'/rest/api/3/issue/{issue_key}', params=params)

            if (response is None):
                failure_msg = "The details of " + str(issue_key) + " are not found"
                print(failure_msg)
                return False, None, failure_msg
                
            else:
                data = response.json()
                #print(data)

                #GET COMMENTS
                comments = ""
                if(getComments):
                    # Assuming 'data' is your JSON object loaded from the response
                    comment_data = data['fields']['comment']['comments']
                    extractcomments = self.extract_comments(comment_data)

                    # Print each comment retrieved
                    for i, comment in enumerate(extractcomments, 1):
                        comments += str(i) + " : " + str(comment) + "\n" 

                #GET ATTACHMENTS
                relevant_attachments = []
                relevant_attachments_links = None
                attachment_list = None
                if(getAttachments):
                    attachment_list = [attachment['filename'] for attachment in data['fields']['attachment']]
                    attachments = data['fields'].get('Attachments', [])
                    
                    # Extract relevant attachments
                    pattern = re.compile(r'CR_.+\.(pdf|docx?|DOCX?)$')
                    for attachment in attachments:
                        if pattern.match(attachment['filename']):
                            relevant_attachments.append(attachment['content']['href'])
                    relevant_attachments_links = ' | '.join(relevant_attachments)  # Join all relevant attachment links with a separator

                #description_content = data.get('fields').get('description', {}).get('content', [])
                #description_text = ' '.join([paragraph['content'][0]['text'] for paragraph in description_content])
                description_content = data.get('fields').get('description', {}).get('content', [])
                description_text = ' '.join([paragraph['content'][0]['text'] for paragraph in description_content if 'text' in paragraph['content'][0]])

                issue_details = {
                    'JIRAID': data['id'],
                    'IssueKey': issue_key,
                    #'UserID': data['fields'].get('assignee', {}).get('accountId', 'No assignee'),
                    'OpenDate': data['fields']['created'],
                    'State': data['fields']['status']['name'],
                    'Close Date': data['fields'].get('resolutiondate', 'Not closed'),
                    'Description': description_text,
                    #'Description': data.get('fields').get('description', 'No description available'),
                    #'Comments': ' | '.join([comment['body'] for comment in data['fields']['comment']['comments']]),
                    'Attachments': attachment_list,
                    'Comments':comments,
                    'RelevantAttachments':relevant_attachments,
                    'RelevantAttachmentsLinks':relevant_attachments_links
                }

                # Print all issue details
                if(printDetails):
                    for key, value in issue_details.items():
                        print(f"{key}: {value}")

                return True, issue_details, None

        except Exception as e:
            failure_msg = f"Fail to retrieve details of " + str(issue_key) + " : {e}"
            print(failure_msg)
            return False, None, failure_msg


        


    

if __name__ == "__main__":
    #RETRIEVE JIRA ID
    app = retrieve_jira()
    success_main_retrieve, issues_dict, issue_amount = app.main_retrieve(project_key = "SCRUM", board_id = None, printIDs=False)
    #app.mainloop()

    if(success_main_retrieve):
        #RETRIEVE JIRA DETAILS
        fields = 'id,assignee,created,status,resolutiondate,description,summary,comment,attachment'
        params={'fields': fields}

        # Open a CSV file to write the issue details
        getComments = True
        with open('jira_issues.csv', mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['IssueID', 'IssueKey', 'User_Story', 'Comments', 'ECCOLA_Code']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for issue_id, issue_key in issues_dict.items():
                #print(f"Issue ID: {issue_id}, Key: {issue_key}")
                success_detail_retrieve, issue_details, failure_msg = app.details_retrieve(issue_key=issue_key, params=params, getComments=getComments, getAttachments=False, printDetails=True)
                if success_detail_retrieve:
                    writer.writerow({
                        'IssueID': issue_id,
                        'IssueKey': issue_key,
                        'User_Story': issue_details.get('Description', ''),
                        #'Comments': issue_details.get('Comments', ''),
                        'Comments': '' if not getComments else issue_details.get('Comments', ''),
                        'ECCOLA_Code': ''  # Empty column
                    })
                else:
                    print(failure_msg)
