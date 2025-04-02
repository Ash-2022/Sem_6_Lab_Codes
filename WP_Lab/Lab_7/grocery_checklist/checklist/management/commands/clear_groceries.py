from django.core.management.base import BaseCommand
from checklist.models import GroceryItem

class Command(BaseCommand):
    help = 'Clears all grocery items from the database'

    def handle(self, *args, **kwargs):
        GroceryItem.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Successfully cleared all grocery items'))
