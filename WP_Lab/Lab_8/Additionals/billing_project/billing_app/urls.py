from django.urls import path
from . import views

urlpatterns = [
    path('', views.page1, name='page1'),
    path('generate_bill/', views.generate_bill, name='generate_bill'),
]
