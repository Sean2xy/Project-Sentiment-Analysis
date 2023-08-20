# -*- coding:utf-8 -*-
import os
import pdb
import re
import json
from flask import Flask, redirect, url_for, request, render_template,jsonify
from flask_cors import CORS
import warnings
import  torch.nn.functional as F
warnings.filterwarnings("ignore")
from transformers import BertTokenizer
import os
import torch


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from model import BertBaseUncased

net = BertBaseUncased()
net.to(device)
weights_path = "./bestModelbase.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
net.load_state_dict(torch.load(weights_path, map_location=device))
id2class = {0: "negative", 1: "positive"}


# test_text.unsqueeze(0) adds a dimension to the input text tensor to fit the input shape of the model. .to(device)
# moves the input tensor to the specified device (e.g. GPU) for computation.
#
# predict_y = torch.max(outputs.logits, dim=1)[1]: gets the index of the predicted category from the model's output.
# outputs.logits is the original output of the model and the torch.max function returns the maximum value of each
# sample and its index, here the index is taken as the predicted category.
#
# probability=outputs.logits[0]: extracts the probability distribution corresponding to the predicted category from
# the output of the model.
#
# probability = F.softmax(probability): use the softmax function to normalise the probability distribution and get
# the probability values for each category.
#
# result = predict_y.cpu().numpy().tolist()[0]: converts the predicted category index to a NumPy array and extracts
# the first element as the predicted result.
#
# Translated with www.DeepL.com/Translator (free version)

def pre(text):
    text_encoded_text = tokenizer.encode_plus(
        text, add_special_tokens=True, truncation=True,
        max_length=128, padding='max_length',
        return_attention_mask=True,
        return_tensors='pt')

    test_text = text_encoded_text['input_ids'][0]
    test_text_mask = text_encoded_text['attention_mask'][0]
    outputs = net(test_text.unsqueeze(0).to(device), test_text_mask.unsqueeze(0).to(device))
    predict_y = torch.max(outputs.logits, dim=1)[1]
    probability=outputs.logits[0]
    probability=F.softmax(probability)
    result = predict_y.cpu().numpy().tolist()[0]
    print(str(id2class[int(result)]))
    return str(id2class[int(result)])+". Probability："+str(float(probability[predict_y].cpu()))


# This code is a simple web application based on the Flask framework for wrapping the prediction function pre(text)
# as an API interface

if __name__ == '__main__':

    app = Flask(__name__)
    CORS(app, supports_credentials=True)
    app.debug = True

    @app.after_request
    def af_request(resp):
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
        resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
        return resp


    @app.route('/', methods=['post'])
    def get_reply():
        if not request.data:
            return ('fail')
        content = request.data.decode('utf-8')
        content_json = json.loads(content)
        user_input = content_json["user_input"]
        print(user_input)
        if len(user_input) < 2:
            return (json.dumps("The empty message or a single character are not allowed. Please type a sentence", ensure_ascii=False))
        reply = pre(user_input)
        return (json.dumps(reply, ensure_ascii=False))


    app.run(host='127.0.0.1',port=8000)