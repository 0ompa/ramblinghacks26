from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload_highlight, name="upload_highlight"),
    path(
        "export/youtube/<str:clip_id>/",
        views.export_youtube_shorts,
        name="export_youtube_shorts",
    ),
]
