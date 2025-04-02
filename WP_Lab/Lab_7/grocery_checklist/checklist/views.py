from django.shortcuts import render
from django.http import JsonResponse
from .models import GroceryItem

def checklist(request):
    items = GroceryItem.objects.all()
    return render(request, 'checklist.html', {'items': items})

def add_item(request):
    if request.method == 'POST':
        item_id = request.POST.get('item_id')
        item = GroceryItem.objects.get(id=item_id)
        return JsonResponse({'name': item.name, 'price': float(item.price)})
