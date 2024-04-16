from transformers import(
    BertTokenizer, 
    BertForQuestionAnswering,
    AutoModelForQuestionAnswering, 
    AutoTokenizer
) 

import torch
from data import DATA

tokenizer = AutoTokenizer.from_pretrained('intel/dynamic_tinybert')
model = AutoModelForQuestionAnswering.from_pretrained('intel/dynamic_tinybert')
def chunk_text(text, max_len=256, overlap=50):
    tokens = tokenizer.tokenize(text)
    chunks = []
    i = 0
    while i < len(tokens):
        start = i
        end = min(i + max_len, len(tokens))
        if end == len(tokens):
            chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
            break
        # Adjust end to avoid breaking words
        shift = len(tokenizer.tokenize(' '.join(tokens[end-overlap:end])))
        end -= shift
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        i += (max_len - overlap)
    return chunks

def answer_question(question, answer_text):
    chunks = chunk_text(answer_text)
    max_score = -float('inf')
    best_answer = None
    
    for chunk in chunks:
        inputs = tokenizer.encode_plus(question, chunk, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        score = (torch.max(answer_start_scores) + torch.max(answer_end_scores)).item()

        if score > max_score:
            max_score = score
            if answer_start <= answer_end:  # Ensure valid span
                answer_tokens = input_ids[answer_start:answer_end]
                extracted_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_tokens))
                if extracted_answer != "[CLS]":
                    best_answer = extracted_answer
                    print(f"New best score found: {max_score} | Answer: {best_answer}")  # Debug: Best score update

    return best_answer

question = "Am I working?"
answer = answer_question(question, DATA)
print("Answer:", answer)