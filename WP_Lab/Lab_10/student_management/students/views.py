from django.shortcuts import render, redirect
from .models import Student
from .forms import StudentForm

def student_view(request):
    if request.method == 'POST':
        form = StudentForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('student_list')
    else:
        form = StudentForm()
    
    students = Student.objects.all()
    return render(request, 'students/student_list.html', {'form': form, 'students': students})
