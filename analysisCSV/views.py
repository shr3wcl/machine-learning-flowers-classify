from django.shortcuts import render, redirect
from .forms import FileUploadForm
from .models import UploadedFile
import pandas as pd
from django.core.paginator import Paginator
from django.http import JsonResponse, HttpResponse
from django.template.loader import render_to_string

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np
import seaborn as sns
import datetime

def index(request):
    form = FileUploadForm()
    uploaded_files = UploadedFile.objects.all()
    return render(request, 'analysisCSV/index.html', {'form': form, 'uploaded_files': uploaded_files})

def upload_file(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save(commit=False)
            if uploaded_file.file.name.endswith('.csv'):
                uploaded_file.save()
                return redirect('upload_success')
            else:
                form.add_error('file', 'Only CSV files are allowed.')
    else:
        return render(request, 'error.html')
        

def upload_success(request):
    return render(request, 'analysisCSV/upload_success.html')

def analyze_csv(request, file_id):
    uploaded_file = UploadedFile.objects.get(pk=file_id)
    df = pd.read_csv(uploaded_file.file)
    explore_data = df.describe()
    explore_html = explore_data.to_html()

    data_preview_html = df.head(10).to_html(index=False)
    
    df_lines = "{:,}".format(len(df))
    paginator = Paginator(df, 10)
    page_number = request.GET.get('page')
    data_all_rows = paginator.get_page(page_number)
    
    columns = df.columns
    
    if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
        return JsonResponse({'table_html': render_to_string('analysisCSV/table.html', {'data_all_rows': data_all_rows})})
    
    return render(request, 'analysisCSV/analyze_csv.html', {'uploaded_file': uploaded_file, 'explore_html': explore_html, 'data_preview_html': data_preview_html, 'df_lines': df_lines, 'data_all_rows': data_all_rows, 'columns': columns, 'file_id': file_id})

def analysis_model(request, file_id):
    if request.method == 'POST':
        target_variable = request.POST.get('target')
        input_columns = request.POST.getlist('input')
        model = request.POST.get('model')
        file_id = request.POST.get('file_id')
        
        result = train_model(model, file_id, target_variable, input_columns)
        if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
            return JsonResponse({'result_form': render_to_string('analysisCSV/analysis_model.html', {"result": result})})
        
    else:
        return render(request, 'analysisCSV/index.html')

def train_model(model, file_id, target, selected_variables):
    uploaded_file = UploadedFile.objects.get(pk=file_id)
    df = pd.read_csv(uploaded_file.file)
    
    df = df.dropna()
    df = df.drop_duplicates()
    categories = []
    nums = []
    for variable in selected_variables:
        if df[variable].dtype == "object":
            categories.append(variable)
        else:
            nums.append(variable)
    df_numericals = df[nums]
    imputer = SimpleImputer(strategy='most_frequent')
    df_categories = df[categories]
    df_categories = pd.DataFrame(imputer.fit_transform(df_categories), columns=df_categories.columns)
    for cate in categories:
        lb_encoder = LabelEncoder()
        df_categories[cate] = lb_encoder.fit_transform(df[cate])
    
    for num in nums:
        mean = df_numericals[num].mean()
        df_numericals.fillna(mean, inplace=True) 
    X = pd.concat([df_numericals, df_categories], axis=1)

    if df[target].dtype == "object":
        lb_encoder = LabelEncoder()
        y = lb_encoder.fit_transform(df[target])
    else:
        y = df[target]

    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    results = {}

    if model == "linear":
        model = LinearRegression()
        model.fit(trainX, trainy)
        y_pred = model.predict(testX)
        rmse = mean_squared_error(testy, y_pred, squared=False)
        results['equation'] = str(model.coef_)  
        results['accuracy'] = f"{rmse:.2f}"  
        results['visualization'] = None       
        return results

    if model == "logistic":
        clf = LogisticRegression()
        clf.fit(trainX, trainy)
        predict = clf.predict(testX)
        accuracy = clf.score(testX, testy)
        report = classification_report(testy, predict)
        cm = confusion_matrix(testy, predict)

        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        random_file_name = uploaded_file.file.name.split("/")[-1].split(".")[0] + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        plt_path = f"static/img/{random_file_name}"
        plt.savefig(plt_path)
        plt.close()

        results = {}
        results['equation'] = None
        results['accuracy'] = f"{accuracy:.2f}"
        results['visualization'] = random_file_name

        return results

    if model == "knn":
        k = 5
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(trainX, trainy)
        y_pred = knn.predict(testX)
        accuracy = accuracy_score(testy, y_pred)
        results['equation'] = None  
        results['accuracy'] = f"{accuracy:.2f}" 
        results['visualization'] = None 
        return results


def serve_file(request, file_name):
    file_path = f'media/uploads/{file_name}'
    with open(file_path, 'rb') as file:
        response = HttpResponse(file.read(), content_type="application/csv")
        response['Content-Disposition'] = f'attachment; filename={file_name}'
        return response

def view_file(request, file_name):
    try:
        file_path = f'media/uploads/{file_name}'
        with open(file_path, 'r') as file:
            response = HttpResponse(file.read(), content_type="text/txt")
            return response
    except:
        return HttpResponse("File not found")

def delete_file(request, file_id):
    uploaded_file = UploadedFile.objects.get(pk=file_id)
    uploaded_file.delete()
    return redirect('index_csv')

def load_image(request, file_name):
    try:
        with open(f"static/img/{file_name}", 'rb') as file:
            response = HttpResponse(file.read(), content_type="image/png")
            return response
    except:
        return HttpResponse("File Not Found")