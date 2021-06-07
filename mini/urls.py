from django.contrib import admin
from django.urls import path
from mini import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
    path('capture', views.capture, name='script'),
]

