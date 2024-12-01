# ECCOLA_AutomatedSuggestion
Automated Suggestion build based on Vakkuri, Ville, et al. | ECCOLA—A method for implementing ethically aligned AI systems | Journal of Systems and Software 182:111067 | 2021

This repo is the code for **proving design decision in subsection 4.6.3 Transformation algorithm** of thesis : https://lutpub.lut.fi/handle/10024/168082

## BART Processing with Vakkuri Dataset
This project demonstrates how to process the Vakkuri Dataset using BART models for natural language processing. Follow the steps below to set up and run the project on your local environment.

## Installation

Before starting, ensure you have Python installed on your system. The author is using VSCode with python3.11 in windows 11.

### Steps to Setup

1. **Clone the Repository**
   - Clone this repository to your local machine using:
     ```bash
     git clone [https://github.com/your-username/your-repository-name.git](https://github.com/cinapr/ECCOLA_AutomatedSuggestion.git)
     cd your-repository-name
     ```

2. **Install Required Libraries**
   - Install the necessary Python libraries by running:
     ```bash
     .\python.exe -m pip install pandas transformers torch pickle
     ```

3. **Get Testing DataSet**
    Testing dataset can be acquired from the [main github repository of this project](https://github.com/cinapr/EthicalAutomatedSuggestion).

### Compare GPT4ALL models
1. Install the depedency:
     ```bash
     pip install gpt4all
     ```
2. Download gguf files that you want to tested from HuggingFace or GPT4ALL website, and place it locally
3. Adjust the gguf files placement on the GPT4ALL_COMPAREMODEL.py
4. Run the python comparison script, you might change the testing questions as well in this script from 'What is London?' to your preference question :
     ```bash
   python GPT4ALL_COMPAREMODEL.py
     ```

### Compare Other LLMs
#### Data Preparation

- Ensure the Vakkuri Dataset is placed in the main directory of the cloned repository.
- For each of the tokenizer and interface you need to ensure the model name and csv name is correct (Check directly on the code)

### Running the Scripts BART
1. **Install depedencies**
   - Install python depedencies:
     ```bash
     pip install pandas transformers torch
     ```

2. **Tokenize the Data**
   - Run the BART tokenizer script:
     ```bash
     python BART_Tokenizer.py
     ```

3. **Perform Inference**
   - After tokenization, run the inference script:
     ```bash
     python BART_Interference.py
     ```

### Running the Scripts GPT2

1. **Install depedencies**
   - Install python depedencies:
     ```bash
     pip install transformers pandas
     ```

2. **Tokenize the Data**
   - Run the GPT2 tokenizer script:
     ```bash
     python GPT2_Tokenizer.py
     ```

3. **Perform Inference**
   - After tokenization, run the inference script:
     ```bash
     python GPT2_Interference.py
     ```

### Running the Scripts GPT4 WITH OLD OPENAI<1.0

1. **Install depedencies**
   - Install python depedencies:
     ```bash
     pip install openai==0.28 pandas
     ```

2. **Tokenize the Data**
   - Run the GPT4 tokenizer script:
     ```bash
     python GPT4_Tokenizer0.py
     ```

3. **Perform Inference**
   - After tokenization, run the inference script:
     ```bash
     python GPT4_Interference0.py
     ```

### Running the Scripts GPT4 WITH OPENAI==1.30.1

1. **Install depedencies**
   - Install python depedencies:
     ```bash
     pip install openai==1.30.1 pandas
     ```

2. **Tokenize the Data**
   - Run the GPT4 tokenizer script:
     ```bash
     python GPT4_Tokenizer1.py
     ```

3. **Perform Inference**
   - After tokenization, run the inference script:
     ```bash
     python GPT4_Interference1.py
     ```


## Contributing

Vakkuri, Ville, et al. "ECCOLA—A method for implementing ethically aligned AI systems." Journal of Systems and Software 182 (2021): 111067.

## License

Distributed under the same license with ECCOLA

## Contact

Your Name - [Cindy Aprilia](www.linkedin.com/in/apriliacindy)

Project Link: [https://github.com/cinapr/ECCOLA_AutomatedSuggestion/](https://github.com/cinapr/ECCOLA_AutomatedSuggestion/)


   
