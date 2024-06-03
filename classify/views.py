from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import FileUploadForm
from requests import post
from .models import DetectFile
from rest_framework.response import Response
from rest_framework import status
import torch
from torch import nn
import torchvision
from torchvision import transforms
import ast
from PIL import Image
from typing import Tuple, List
import numpy as np
from queue import Queue
from threading import Thread
import time
import cv2
from ultralytics import YOLO
from .detect import predict_all_flowers
import os

model = YOLO("models/best_flower_2.onnx")

# Create your views here.
def index(request):
    form = FileUploadForm()
    uploaded_files = DetectFile.objects.all()
    uploaded_files = [file.file.name.split('/')[-1] for file in uploaded_files]
    return render(request, 'classify/index.html', {'form': form, 'images': uploaded_files})

# def detect(request):
#     if request.method == 'POST':
#         form = FileUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#             URL = "http://localhost:8000/plants/detect2/"
#             files = {'files': open('media/images/' + str(request.FILES['file']), 'rb')}
#             response = post(URL, files=files)
#             # Print result in response to console
#             print(response.json())
#             return render(request, 'classify/result.html', {'image_url': str(request.FILES['file']), 'result': response.json().get('result'), 'status': response.json().get('status')})
#         else:
#             print(form.errors)
    
def image_view(request, file_name):
    try:
        with open(f"media/images/{file_name}", 'rb') as file:
            response = HttpResponse(file.read(), content_type="image/png")
            return response
    except:
        return HttpResponse("File Not Found")

def detect(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            input2 = request.FILES.get('file')
            print("Name image:",input2)
            temp_file_path = "image.jpg"
            with open(temp_file_path, 'wb') as temp_file:
                for chunk in input2.chunks():
                    temp_file.write(chunk)

            image_flower = cv2.imread(temp_file_path)
            image_flower = cv2.resize(image_flower, (224, 224))
            
            results = model(image_flower, stream=True)
            
            bboxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

                    bboxes.append((x1, y1, x2, y2))

                results = predict_all_flowers(bboxes=bboxes, image_original=image_flower)
                print(results)
                for result in results:
                    pred_label_decoded, conf, x1, y1, x2, y2 = result
                    cv2.rectangle(image_flower, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(image_flower, pred_label_decoded, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite("media/images/output.jpg", image_flower)
            
            labels = []
            for result in results:
                pred_label_decoded, conf, x1, y1, x2, y2 = result
                labels.append(pred_label_decoded)
            os.remove(temp_file_path)

            return render(request, 'classify/result.html', {'image_url': str(request.FILES['file']), 'result': labels, 'status': "success", 'img_result': 'output.jpg'})
        
