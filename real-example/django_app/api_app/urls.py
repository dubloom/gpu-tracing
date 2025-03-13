from django.urls import path
from .views import call_flask

urlpatterns = [
    path('call_flask/', call_flask),  # Expose this endpoint
]