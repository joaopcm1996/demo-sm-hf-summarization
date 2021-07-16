
# This is the script that will be used in the inference container
import os 
import json 
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def model_fn(model_dir):
    """
    Load the model and tokenizer for inference 
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    
    model_dict = {'model':model, 'tokenizer':tokenizer}
    
    return model_dict 


def predict_fn(input_data, model):
    """
    Make a prediction with the model
    """
    

    text = input_data.pop('inputs')
    parameters = input_data.pop('parameters', None)
    
    tokenizer = model['tokenizer']
    model = model['model']

    # Parameters may or may not be passed    
    input_ids = tokenizer(text, truncation=True, padding='longest', return_tensors="pt").input_ids
    output = model.generate(input_ids, **parameters) if parameters is not None else model.generate(input_ids)
    
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


def input_fn(request_body, request_content_type):
    """
    Transform the input request to a dictionary
    """
    request = json.loads(request_body)

    return request


def output_fn(prediction, response_content_type):
    """
    Return model's prediction
    """
    return {'generated_text':prediction}
