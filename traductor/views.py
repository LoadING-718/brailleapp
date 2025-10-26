from django.shortcuts import render
# translator/views.py
from django.shortcuts import render, redirect
from .forms import BrailleImageForm
from .models import BrailleImage
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

BRAILLE_DICT = {
    (1,0,0,0,0,0): 'A', (1,1,0,0,0,0): 'B', (1,0,0,1,0,0): 'C', (1,0,0,1,1,0): 'D',
    (1,0,0,0,1,0): 'E', (1,1,0,1,0,0): 'F', (1,1,0,1,1,0): 'G', (1,1,0,0,1,0): 'H',
    (0,1,0,1,0,0): 'I', (0,1,0,1,1,0): 'J', (1,0,1,0,0,0): 'K', (1,1,1,0,0,0): 'L',
    (1,0,1,1,0,0): 'M', (1,0,1,1,1,0): 'N', (1,0,1,0,1,0): 'O', (1,1,1,1,0,0): 'P',
    (1,1,1,1,1,0): 'Q', (1,1,1,0,1,0): 'R', (0,1,1,1,0,0): 'S', (0,1,1,1,1,0): 'T',
    (1,0,1,0,0,1): 'U', (1,1,1,0,0,1): 'V', (0,1,0,1,1,1): 'W', (1,0,1,1,0,1): 'X',
    (1,0,1,1,1,1): 'Y', (1,0,1,0,1,1): 'Z'
}

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def detect_centers(thresh, min_size=8, max_size=200):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if min_size < w < max_size and min_size < h < max_size:
            centers.append((x + w/2.0, y + h/2.0))
    return centers


def cluster_cells(centers, eps=40, min_samples=1):
    """Agrupa centros en clusters (una celda ≈ un cluster) usando DBSCAN."""
    if not centers:
        return []
    X = np.array(centers)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(X)
    labels = db.labels_  # -1 = noise
    clusters = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        pts = X[labels == lab].tolist()
        clusters.append(pts)
    # Ordenar clusters por coordenada x (de izquierda a derecha)
    clusters.sort(key=lambda c: np.mean([p[0] for p in c]))
    return clusters

def cell_pattern_from_cluster(cluster):
    """
    Traduce una celda (lista de puntos) a un patrón Braille (6 bits)
    considerando inversión vertical y columnas izquierda/derecha.
    """
    if len(cluster) < 1:
        return (0,0,0,0,0,0)

    xs = np.array([p[0] for p in cluster])
    ys = np.array([p[1] for p in cluster])

    # Normaliza coordenadas
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    w, h = maxx - minx, maxy - miny

    # Invertimos eje Y para que el origen esté arriba
    ys = maxy - ys

    # Umbrales dinámicos para filas (3) y columnas (2)
    col_threshold = (minx + maxx) / 2
    row_thresholds = [miny + h/3, miny + 2*h/3]

    pattern = [0]*6
    for (x, y) in zip(xs, ys):
        col = 0 if x < col_threshold else 1
        row = 0 if y > row_thresholds[1] else (1 if y > row_thresholds[0] else 2)
        # Índice Braille estándar: 1–3 col izquierda, 4–6 col derecha
        idx = col*3 + row
        if 0 <= idx < 6:
            pattern[idx] = 1

    return tuple(pattern)


def translate_braille(centers):
    """
    pipeline: detect_centers -> cluster_cells -> cell_pattern_from_cluster -> map
    """
    clusters = cluster_cells(centers, eps=10, min_samples=1)
    text = ""
    for cl in clusters:
        pat = cell_pattern_from_cluster(cl)
        text += BRAILLE_DICT.get(pat, '?')
    return text

def translate_image(image_path):
    thresh = preprocess_image(image_path)
    centers = detect_centers(thresh)
    clusters = cluster_cells(centers, eps=20, min_samples=1)

    # Visualización (solo depuración)
    vis = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    for i, cl in enumerate(clusters):
        color = colors[i % len(colors)]
        for (x, y) in cl:
            cv2.circle(vis, (int(x), int(y)), 10, color, 2)
    cv2.imwrite("clusters_debug.jpg", vis)

    translated = ""
    for cl in clusters:
        pat = cell_pattern_from_cluster(cl)
        print("Patron detectado", pat)
        translated += BRAILLE_DICT.get(pat, '?')
    return translated

#def translate_image(image_path):
#    thresh = preprocess_image(image_path)
#    centers = detect_centers(thresh)
#    translated = translate_braille(centers)
#    return translated

"""
def detect_braille_dots(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if 5 < w < 50 and 5 < h < 50:           # Para filtrar ruido
            centers.append((x + w/2, y + h/2))
    centers.sort(key=lambda p: (p[1], p[0]))    # Ordena por fila
    return centers

def group_into_cells(centers, cell_width=60, cell_height=90):
    #Agrupa puntos en celdas de 2x3 suponiendo espaciado uniforme.
    if not centers:
        return []
    centers = sorted(centers, key=lambda p: (p[0], p[1]))
    min_x = min(c[0] for c in centers)
    max_x = max(c[0] for c in centers)
    num_cells = int((max_x - min_x) // cell_width) + 1

    cells = [[] for _ in range(num_cells)]
    for (x, y) in centers:
        idx = int((x - min_x) // cell_width)
        cells[idx].append((x, y))
    return cells

def translate_braille(dots):
    text = ""
    for cell in dots:
        # Ordenar puntos dentro de celda por posición (fila,columna)
        cell = sorted(cell, key=lambda p: (p[0], p[1]))
        # Determinar posiciones relativas (simplificación)
        pattern = [0]*6
        for (x, y) in cell:
            # Determinar punto según su posición vertical y horizontal
            col = 0 if (x % 120) < 60 else 1
            row = int((y % 180) // 30)
            idx = col * 3 + row
            if 0 <= idx < 6:
                pattern[idx] = 1
        pattern_tuple = tuple(pattern)
        text += BRAILLE_DICT.get(pattern_tuple, '?')
    return text

def translate_image(image_path):
    thresh = preprocess_image(image_path)
    centers = detect_braille_dots(thresh)
    cells = group_into_cells(centers)
    translated_text = translate_braille(cells)
    return translated_text
"""
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
    return render(request, 'traductor/result.html', {'braille_image': braille_image})

