import re
from bs4 import BeautifulSoup

from django.utils.html import escape
from django.views.generic import View
from django.shortcuts import render, redirect
from django.http import HttpResponse

from .utils import first_ranker, second_ranker, dataset


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
        first_results = first_ranker(query)
        second_results = second_ranker(first_results, query)
        docs = [
            (
                docindex,
                dataset.docs[docindex].title
            )
            for docindex in second_results
        ]

        context = {'query': query, 'results': docs}
        return render(request, 'doogle_results.html', context)
    
    def post(self, request):
        document_id = request.POST.get('document_id')
        return redirect('doogle_document', document_id=document_id)


class DocumentPageView(View):
    url_name = 'doogle_document'
    def get(self, request, document_id):
        doc = dataset.docs[int(document_id)]

        # with open(f'AP_collection/coll/{doc_file_name}', 'r', encoding='utf-8') as file:
        #     file_content = file.read()

        # soup = BeautifulSoup(file_content, 'html.parser')

        document = f"""
        <html>
            <head>
                <title>{escape(doc.title)}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                    }}
                    h1 {{
                        color: #2a3d66;
                    }}
                    p {{
                        line-height: 1.6;
                    }}
                    .document-url {{
                        font-size: 0.9em;
                        color: #666;
                        margin-top: 20px;
                    }}
                </style>
            </head>
            <body>
                <h1>{escape(doc.title)}</h1>
                <p>{escape(doc.text)}</p>
                <p class="document-url"><strong>URL:</strong> <a href="{escape(doc.url)}">{escape(doc.url)}</a></p>
            </body>
        </html>
        """
        return HttpResponse(document, content_type='text/html')

