from django.shortcuts import render
# translator/views.py
from django.shortcuts import render, redirect
from .forms import BrailleImageForm
from .models import BrailleImage
import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    return thresh

def detect_braille_dots(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    braille_dots = [cv2.boundingRect(c) for c in contours]
    return braille_dots

def translate_braille(dots):
    braille_text = ""
    for dot in dots:
        braille_text += "."
    return braille_text

def translate_image(image_path):
    thresh = preprocess_image(image_path)
    braille_dots = detect_braille_dots(thresh)
    translated_text = translate_braille(braille_dots)
    return translated_text

def upload_image(request):
    if request.method == 'POST':
        form = BrailleImageForm(request.POST, request.FILES)
        if form.is_valid():
            braille_image = form.save()
            image_path = braille_image.image.path
            translated_text = translate_image(image_path)
            braille_image.translated_text = translated_text
            braille_image.save()
            return redirect('result', pk=braille_image.pk)
    else:
        form = BrailleImageForm()
    return render(request, 'traductor/upload.html', {'form': form})

def result(request, pk):
    braille_image = BrailleImage.objects.get(pk=pk)
    return render(request, 'traducctor/result.html', {'braille_image': braille_image})

