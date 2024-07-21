from gpt4all import GPT4All
import time
import os

models = [
    "D:\\SE4GD\\Semester4\\Code\\GPT4ALL\\mistral-7b-openorca.gguf2.Q4_0.gguf",
    "D:\\SE4GD\\Semester4\\Code\\GPT4ALL\\Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
    "D:\\SE4GD\\Semester4\\Code\\GPT4ALL\\orca-mini-3b-gguf2-q4_0.gguf",
    "D:\\SE4GD\\Semester4\\Code\\GPT4ALL\\replit-code-v1_5-3b-newbpe-q4_0.gguf"
]

for model_path in models:
    try:
        start_time = time.time()
        device = 'cuda:0'  # Make sure this index corresponds to your NVIDIA GPU
        model = GPT4All(model_path, device=device)
        output = model.generate("The capital of France is ", max_tokens=1003)
        print(output)
        end_time = time.time()
        model_filename = os.path.basename(model_path)
        print(f"Time taken for {model_filename}: {end_time - start_time:.2f} seconds")
    except Exception as e:
        model_filename = os.path.basename(model_path)
        print(f"Error during generation with {model_filename}:", e)
