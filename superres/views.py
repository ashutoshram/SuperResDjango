# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse

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

model_path = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(model_path, "model_4x4_3x3_clamp.pth")

output_filename_url = ''


def run_super_res(uploaded_file, model_filename, output_filename):
    t = threading.Thread(target=super_res, args=(uploaded_file, model_filename, output_filename))
    t.start()


def home(request):
    documents = Document.objects.all()
    return render(request, 'home.html', { 'documents': documents })

def super_res(input_filename, model_filename, output_filename):
   cuda = False 
   here = os.path.dirname(os.path.abspath(__file__))
   sys.path.append(here)
   model_ = torch.load(model_filename)
   super_resolve.super_resolve(input_filename, model_, output_filename, cuda)
   global output_filename_url
   #fs = FileSystemStorage()
   output_filename_url = os.path.join(STATIC_URL, os.path.basename(output_filename))
   #output_filename_url = fs.url(os.path.basename(output_filename))
   print('super_res: output_filename_url = ', output_filename_url)
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
        print('output_filename = %s' % output_filename)
        output_filename = os.path.join(STATICFILES_DIRS[0], output_filename)
        print('output_filename(after adding static) = %s' % output_filename)
        run_super_res(abs_uploaded_file_path, model_filename, output_filename)
        #super_res calls the super-resolution on the uploaded file and saves it to the output file. Returns width and height.
       
        global output_filename_url
        output_filename_url = ""
        
        ongoing = True
        
        print('simple_upload: returning response page')
        return render(request, 'simple_upload_response.html', {
            'ongoing': ongoing,
            'uploaded_file_url': uploaded_file_url,
        })
    print('simple_upload: returning default page')
    return render(request, 'simple_upload.html')


def query_status(request):
   global output_filename_url
   print('query_status: output_filename_url = ', output_filename_url)
   data = {
        'status': 'in progress' if output_filename_url == "" else 'done',
        'image_url': output_filename_url
    }
   print('got a query_status, sending respose ', data)
   return JsonResponse(data)
   
