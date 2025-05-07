from django.urls import path
from .views import search_api

urlpatterns = [
    path('chat/', search_api, name='search_api'),  # buang 'api/' di sini
]
