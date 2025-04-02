from django.shortcuts import render, get_object_or_404
from django.views import generic
from .models import Institutes

def institutes_list(request):
    institutes = Institutes.objects.all()
    return render(request, 'institutes_app/institutes_list.html', {'institutes': institutes})

class InstituteDetailView(generic.DetailView):
    model = Institutes
    template_name = 'institutes_app/institute_detail.html'
    context_object_name = 'institute'
