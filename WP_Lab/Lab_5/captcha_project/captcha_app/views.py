from django.shortcuts import render

def captcha_view(request):
    return render(request, 'captcha_app/index.html')
