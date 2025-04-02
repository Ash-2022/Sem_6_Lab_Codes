# institutes_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.institutes_list, name='institutes_list'),
    path('institute/<int:pk>/', views.InstituteDetailView.as_view(), name='institute-detail'),
]
