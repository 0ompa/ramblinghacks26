"""
URL configuration for highlights_site project.
"""

from django.conf import settings
from django.contrib import admin
from django.urls import include, path, re_path
from django.views.static import serve

urlpatterns = [
    path("admin/", admin.site.urls),
    # Must be before path("", include(...)) so /videos/* is not routed to clips.urls
    re_path(
        r"^videos/(?P<path>.*)$",
        serve,
        {"document_root": str(settings.VIDEOS_DIR)},
    ),
    path("", include("clips.urls")),
]
