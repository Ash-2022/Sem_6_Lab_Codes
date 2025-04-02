from django.db import models
from django.urls import reverse

class Institutes(models.Model):
    institute_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    no_of_courses = models.IntegerField()
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        return reverse('institute-detail', args=[str(self.institute_id)])
