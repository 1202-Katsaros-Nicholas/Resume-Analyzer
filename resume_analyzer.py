"""
Created by: Nicholas Katsaros
Creation date: April 13, 2024
Purpose: To analyze resumes using BERT and GPT2
"""

import torch
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
from scipy.spatial.distance import cosine
import docx2txt


def generate_job_skills(job_name):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_text = "Skills required for a " + job_name + " position include"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        early_stopping=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


def file_to_string(filepath):
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r') as file:
                contents = file.read()
        elif filepath.endswith('.docx'):
            contents = docx2txt.process(filepath)
        else:
            return "Unsupported file format."

        return contents
    except FileNotFoundError:
        return "File not found."


def preprocess_text(text, tokenizer, max_length=512):
    # Tokenizes and truncates the text to fit the maximum length
    # supported by the BERT model.
    tokenized_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    return tokenized_text


def calculate_similarity(embeddings1, embeddings2):
    # Calculates cosine similarity between two sets of embeddings.
    embeddings1 = embeddings1.flatten()
    embeddings2 = embeddings2.flatten()
    return 1 - cosine(embeddings1, embeddings2)


def main():
    # Load BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Input
    resume = input("Enter the filepath of the resume: ")
    resume_text = file_to_string(resume)
    if resume_text == "File not found.":
        print(resume_text)
        return 1

    job_title = input("Enter the name/title of the position: ")

    # Tokenize text
    tokenized_resume = preprocess_text(resume_text, tokenizer)
    tokenized_job_description = preprocess_text(generate_job_skills(job_title), tokenizer)

    # Compute embeddings
    with torch.no_grad():
        resume_embeddings = model(**tokenized_resume)[0][:, 0, :].numpy()
        job_embeddings = model(**tokenized_job_description)[0][:, 0, :].numpy()

    # Calculate similarity
    similarity_score = calculate_similarity(resume_embeddings, job_embeddings)

    if similarity_score < 0.6:
        print(f"This candidate seems to be a poor fit for the given position.")
    elif similarity_score < 0.7:
        print(f"This candidate is a decent fit for the given position.")
    else:
        print(f"This is a great candidate for this position!")
    print(f"Compatibility Score: {similarity_score}")


if __name__ == "__main__":
    main()
