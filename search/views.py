import re
from bs4 import BeautifulSoup

from django.views.generic import View
from django.shortcuts import render, redirect
from django.http import HttpResponse

from .utils import search_query


# Create your views here.


class SearchPageView(View):
    url_name = 'doogle_search'

    def get(self, request):
        return render(request, 'doogle_search.html')

    def post(self, request):
        query = request.POST.get('q')
        return redirect('doogle_results', query=query)


class ResultPageView(View):
    url_name = 'doogle_results'

    def get(self, request, query):
        print(query)
        results = search_query(query)
        context = {'query': query, 'results': results}
        return render(request, 'doogle_results.html', context)
    
    def post(self, request):
        document_id = request.POST.get('document_id')
        return redirect('doogle_document', document_id=document_id)


class DocumentPageView(View):
    url_name = 'doogle_document'
    def get(self, request, document_id):
        doc_file_name, doc_num_str = document_id.split('-')
        doc_num_int = int(doc_num_str) -1

        with open(f'AP_collection/coll/{doc_file_name}', 'r', encoding='utf-8') as file:
            file_content = file.read()

        soup = BeautifulSoup(file_content, 'html.parser')

        doc_tags = soup.find_all('doc')
        document = doc_tags[doc_num_int]
        return HttpResponse(document, content_type='text/plain')

