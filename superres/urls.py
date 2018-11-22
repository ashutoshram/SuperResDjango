from django.conf.urls import url, include

from . import views

urlpatterns = [
    url(r'^$', views.simple_upload, name='simple_upload'),
    url(r'^query_status/$', views.query_status, name='query_status'),
]
