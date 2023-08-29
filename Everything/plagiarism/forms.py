from django import forms

class CheckForm(forms.Form):
    bio_1 = forms.CharField(widget=forms.Textarea(attrs={'rows': 5}))
    bio_2 = forms.CharField(widget=forms.Textarea(attrs={'rows': 5}))