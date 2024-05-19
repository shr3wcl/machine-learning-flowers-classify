from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import FileUploadForm
from requests import post
from .models import DetectFile

# Create your views here.
def index(request):
    form = FileUploadForm()
    uploaded_files = DetectFile.objects.all()
    uploaded_files = [file.file.name.split('/')[-1] for file in uploaded_files]
    return render(request, 'classify/index.html', {'form': form, 'images': uploaded_files})

def detect(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            URL = "http://localhost:8000/plants/detect2/"
            files = {'files': open('media/images/' + str(request.FILES['file']), 'rb')}
            response = post(URL, files=files)
            # Print result in response to console
            print(response.json())
            return render(request, 'classify/result.html', {'image_url': str(request.FILES['file']), 'result': response.json().get('result'), 'status': response.json().get('status')})
        else:
            print(form.errors)
    
def image_view(request, file_name):
    try:
        with open(f"media/images/{file_name}", 'rb') as file:
            response = HttpResponse(file.read(), content_type="image/png")
            return response
    except:
        return HttpResponse("File Not Found")
