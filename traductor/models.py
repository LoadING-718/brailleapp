from django.db import models

# Create your models here.

class BrailleImage(models.Model):
    image = models.ImageField(upload_to='braille_images/')
    translated_text = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)