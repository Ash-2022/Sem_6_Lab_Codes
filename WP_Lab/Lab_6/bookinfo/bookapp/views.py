from django.shortcuts import render

def home(request):
    return render(request, 'bookapp/home.html')

def metadata(request):
    return render(request, 'bookapp/metadata.html')

def reviews(request):
    return render(request, 'bookapp/reviews.html')

def publisher_info(request):
    return render(request, 'bookapp/publisher_info.html')
