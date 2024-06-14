from django import forms
from .models import BrailleImage

class BrailleImageForm(forms.ModelForm):
    class Meta:
        model = BrailleImage
        fields = ['image']