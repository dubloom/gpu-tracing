import requests
import os
from django.http import JsonResponse

FLASK_API_URL = os.getenv("FLASK_API_URL", "http://flask:5000")

def call_flask(request):
    try:
        response = requests.get(f"{FLASK_API_URL}/matmul")
        return JsonResponse(response.json())
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
