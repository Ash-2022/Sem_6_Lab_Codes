# feedback/views.py
from django.shortcuts import render, redirect
from .forms import FeedbackForm

def feedback_view(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            # Process form data here
            return render(request, 'feedback/thankyou.html')
    else:
        form = FeedbackForm()
    return render(request, 'feedback/form.html', {'form': form})
