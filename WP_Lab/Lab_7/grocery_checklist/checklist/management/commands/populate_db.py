from django.core.management.base import BaseCommand
from checklist.models import GroceryItem

class Command(BaseCommand):
    help = 'Populates the database with initial grocery items'

    def handle(self, *args, **kwargs):
        GroceryItem.objects.create(name="Apples", price=2.99)
        GroceryItem.objects.create(name="Bread", price=1.99)
        GroceryItem.objects.create(name="Milk", price=3.49)
        GroceryItem.objects.create(name="Eggs", price=2.79)

        self.stdout.write(self.style.SUCCESS('Successfully populated the database'))
