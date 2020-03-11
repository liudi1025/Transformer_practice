from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import bs4
from bs4 import BeautifulSoup as BS
import requests
import pandas as pd

import time

def getHtml(url):
    return requests.get(url).text

def parse(html):
    html = getHtml(html)
    soup = BS(html,'html.parser',exclude_encodings='utf-8')
    links = soup.find_all('p')
    links.extend(soup.find_all('span'))
    res = set()
    for link in links:
        txt = link.text.replace('\n','').replace('\xa0','').strip()
        if len(txt.split()) > 10:   # a paragh: more than 10 words
            res.add(txt)
    return list(res)

def replace_symbol(x):
    x = str(x)
    x=x.replace('\r','')
    x=x.replace('\t','')
    x=x.replace('\n','')
    return x

def multiURL_df(url_lst=['http://www.dpgllc.com/site/']*100):
    paraghs = []
    for url in url_lst:
        try:
            paraghs.extend(parse(url))
        except:
            continue
    return paraghs

def get_url_lst(file_path='./html_text/url.csv'):
    '''
    inputs:
        file_path: list of urls
    Returns:
    '''
    url_lst = pd.read_csv(file_path)['url'].tolist()
    return url_lst

def main():
    ## obtain url list
    # paraghs = multiURL_df(get_url_lst())  # from db.tbl
    time_1 = time.time()
    paraghs = multiURL_df()
    # for paragh in paraghs:
    #     print(paragh)
    time_2 = time.time()

    ## load fine-tune model
    tokenizer = AutoTokenizer.from_pretrained("/data1/liudi/Transformers/examples/output/desc")
    model = AutoModelForSequenceClassification.from_pretrained("/data1/liudi/Transformers/examples/output/desc")
    time_3 = time.time()
    desc = tokenizer.batch_encode_plus(paraghs, return_tensors="pt")
    desc_classification_logits = model(**desc)[0]
    results = torch.softmax(desc_classification_logits, dim=1).tolist()
    #print(results)
    time_4 = time.time()

    print(time_2-time_1, time_3-time_2, time_4-time_3)
    ## extract the best description paragraph
    prob, desc_idx = 0, 0
    for i in range(len(paraghs)):
        if results[i][1] > prob:
            prob = results[i][1]
            desc_idx = i
    if prob == 0:
        print('N/A')
        return 'N/A'
    print(paraghs[desc_idx])
    return paraghs[desc_idx]

if __name__ == '__main__':
    main()