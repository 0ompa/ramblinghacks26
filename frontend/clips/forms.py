from django import forms


class HighlightUploadForm(forms.Form):
    video = forms.FileField(
        label="Highlight MP4",
        help_text="Landscape highlight; portrait crop is generated automatically.",
    )

    def clean_video(self):
        f = self.cleaned_data["video"]
        name = (f.name or "").lower()
        if not name.endswith(".mp4"):
            raise forms.ValidationError("Please upload an .mp4 file.")
        return f
