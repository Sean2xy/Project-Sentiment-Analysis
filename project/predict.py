
import warnings
warnings.filterwarnings("ignore")
from transformers import BertTokenizer,RobertaTokenizer
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
id2class={0:"消极",1:"积极"}


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
    result = predict_y.cpu().numpy().tolist()[0]
    print(str(id2class[int(result)]))
