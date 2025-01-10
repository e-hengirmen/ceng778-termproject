
from django.contrib import admin
from django.urls import path, include, re_path
from .views import *

urlpatterns = [
    path(
        '',
        SearchPageView.as_view(),
        name=SearchPageView.url_name
    ),
    re_path(
        r'results/(?P<query>.+)',
        ResultPageView.as_view(),
        name=ResultPageView.url_name
    ),
    re_path(
        r'document/(?P<document_id>.+)',
        DocumentPageView.as_view(),
        name=DocumentPageView.url_name
    ),
]
