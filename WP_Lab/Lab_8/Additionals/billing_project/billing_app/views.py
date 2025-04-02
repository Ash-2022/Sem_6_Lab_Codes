from django.shortcuts import render
from django.http import HttpResponse

def page1(request):
    if request.method == 'GET':
        return render(request, 'billing_app/page1.html')

def generate_bill(request):
    if request.method == 'POST':
        brand = request.POST.get('brand', 'Unknown')
        items = request.POST.getlist('item')
        quantity = int(request.POST.get('quantity', 0))

        # Calculate total amount (example logic: $100 per item per quantity)
        total_amount = 100 * len(items) * quantity

        context = {
            'brand': brand,
            'items': items,
            'quantity': quantity,
            'total_amount': total_amount
        }
        return render(request, 'billing_app/page2.html', context)
