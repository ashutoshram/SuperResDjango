# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from urllib2 import quote

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from superres.models import URLToken

#from uploads.core.models import Document
#from uploads.core.forms import DocumentForm
from resolve.settings import *
import os
import numpy as np
import urllib
import torch
import cv2
from PIL import Image
import sys
import threading
import super_resolve
import time
import uuid

model_path = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(model_path, "model_4x4_3x3_clamp.pth")


def run_super_res(uploaded_file, model_filename, output_filename, token_given):
    t = threading.Thread(target=super_res, args=(uploaded_file, model_filename, output_filename, token_given))
    t.start()


def home(request):
    documents = Document.objects.all()
    return render(request, 'home.html', { 'documents': documents })

def super_res(input_filename, model_filename, output_filename, token_given):
   cuda = False 
   here = os.path.dirname(os.path.abspath(__file__))
   sys.path.append(here)
   model_ = torch.load(model_filename)
   super_resolve.super_resolve(input_filename, model_, output_filename, cuda)
  
   record = URLToken.objects.get(token=token_given)
   record.status = True
   print("Updating record with url %s and token %s with status True" % (record.token, record.url))
   record.save()
   return True
   
   #return url of saved output_image
def genoutputfilename(input_filename):
   base, extension = os.path.splitext(input_filename)
   return os.path.basename(base) + '_superres' + extension

def simple_upload(request):

        
    if request.method == 'POST' and request.FILES['myfile']:

        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        
        abs_uploaded_file_path = os.path.join(BASE_DIR, uploaded_file_url[:]) # remove the / in front of uploaded_file_url
        # launch a thread to run super-res on filename
            
        output_filename = genoutputfilename(abs_uploaded_file_path)
        output_filename = os.path.join(STATICFILES_DIRS[0], output_filename)
        imageurl = os.path.join(STATIC_URL, os.path.basename(output_filename))
        token_given = str(uuid.uuid4())
        print('simple_upload token: ', token_given)
        record = URLToken(url=imageurl, token=token_given, status=False)
        record.save() 
        print("Creating record with url %s and token %s with status False" % (record.token, record.url))
        run_super_res(abs_uploaded_file_path, model_filename, output_filename, token_given)
        #super_res calls the super-resolution on the uploaded file and saves it to the output file. Returns width and height.
       
        
        print('simple_upload: returning response page')
        return render(request, 'simple_upload_response.html', {
            'uploaded_file_url': uploaded_file_url,
            'token': token_given,
        })
    print('simple_upload: returning default page')
    return render(request, 'simple_upload.html')


def query_status(request):
   
   #print('query_status: output_filename_url = ', output_filename_url)
   token_database = URLToken.objects.all()
   incoming_token = request.META['QUERY_STRING']

   print('query status token from ajax =', incoming_token)
   status = False
   image_url = ''
   if token_database.get(token=incoming_token).status:
      image_url = token_database.get(token=incoming_token).url
      image_url = quote(image_url.encode('utf-8'))
      print('query status image_url:', image_url)
      status = True
         
         

   data = {
      'status': 'done' if status else 'in progress',
      'image_url': image_url  
   }
   print('got a query_status, sending response ', data)
   return JsonResponse(data)

