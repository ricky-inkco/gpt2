
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from multiprocessing import Process
import subprocess
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import json
import fire
import torch
from urllib.parse import urlparse, unquote
from flask import Flask, request,jsonify



model: RobertaForSequenceClassification = None
tokenizer: RobertaTokenizer = None
device: str = None

from flask import Flask

app = Flask(__name__)

@app.route('/predict/', methods=['GET', 'POST'])
def do_predict():
    data = json.loads(request.data)
    text = data.get("my_text",None)
    prediction=predict_func(text)
    return jsonify({"fake":prediction[0],"real":prediction[1]})
    
   

def predict_func(query):
    data = torch.load('detector-base.pt', map_location='cpu')
    model_name = 'roberta-base' 
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    model.load_state_dict(data['model_state_dict'])
    model.eval()

    tokens = tokenizer.encode(query)
    all_tokens = len(tokens)
    tokens = tokens[:tokenizer.max_len - 2]
    used_tokens = len(tokens)
    tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
    mask = torch.ones_like(tokens)

    with torch.no_grad():
        logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
        probs = logits.softmax(dim=-1)

    fake, real = probs.detach().cpu().flatten().numpy().tolist()
    return fake,real

if __name__ == '__main__':
    #print(predict("this text needs to be predicted"))
    app.run(host='0.0.0.0', port=8080)