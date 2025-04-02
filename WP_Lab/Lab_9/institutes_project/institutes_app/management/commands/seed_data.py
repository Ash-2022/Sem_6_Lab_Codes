# institutes_app/management/commands/seed_data.py
from django.core.management.base import BaseCommand
from institutes_app.models import Institutes

class Command(BaseCommand):
    help = 'Seed data for Institutes model'

    def handle(self, *args, **kwargs):
        institutes_data = [
            {'name': 'Harvard University', 'no_of_courses': 200},
            {'name': 'MIT', 'no_of_courses': 180},
            {'name': 'Stanford University', 'no_of_courses': 190},
            {'name': 'Oxford University', 'no_of_courses': 150},
            {'name': 'Cambridge University', 'no_of_courses': 160},
        ]
        
        for institute in institutes_data:
            Institutes.objects.create(
                name=institute['name'],
                no_of_courses=institute['no_of_courses']
            )
            
        self.stdout.write(self.style.SUCCESS('Successfully seeded institutes data'))
