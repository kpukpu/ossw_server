from multiprocessing import context
from django.shortcuts import render, HttpResponse
import requests
from bs4 import BeautifulSoup

from transformers import BertTokenizer #기사 요약용
from transformers import BertModel#기사 요약용
from transformers import AutoModel, AutoTokenizer#기사 요약용
import torch
from summarizer import Summarizer#기사 요약용

# Create your views here.

newscontext = " "

def index(request):
    return render(request, 'runningpy/base.html')


def calculate(request):
    try:
        global newscontext
        if request.method == 'POST':
            n_link= request.POST.get('n1')
        # HTTP GET 요청을 보내고 응답을 받음
        response = requests.get(n_link) 
        response.raise_for_status()  # 오류가 있을 경우 예외 발생

        # BeautifulSoup 객체 생성, HTML 파서로 'html.parser' 사용
        soup = BeautifulSoup(response.text, 'html.parser')

        # 'id'가 'dic_area'인 <article> 태그 찾기
        content_article = soup.find('article', id='dic_area')
        # <article> 태그 내의 모든 텍스트 추출
        news_content = ' '.join(content_article.stripped_strings)
        newscontext = news_content
        return render(request, 'runningpy/crawl.html', {'news_content': news_content})
    except requests.RequestException as e:
        return f"HTTP Error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

'''
def calculate(request):
    if request.method == 'POST':
        number1 = request.POST.get('n1')
        
    return HttpResponse('결과 : ' + str(result))
'''
#def summarize(request):
    
def result(request):
    return HttpResponse('''''')


def summarize(request):
    # 모델과 토크나이저 초기화
    model_name = "google/bert_uncased_L-4_H-256_A-4"  # BERT-Mini 모델
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)


    # 텍스트를 모델 입력 형식으로 인코딩
    encoded_input = tokenizer(newscontext, return_tensors='pt')

    # Forward pass, 결과 추출
    with torch.no_grad():
        output = model(**encoded_input)

    model2 = Summarizer()
    summar = model2(newscontext)
    return HttpResponse(summar)