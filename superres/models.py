# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.

class URLToken(models.Model):
   token = models.CharField(max_length=30)
   url = models.CharField(max_length=30)
   status = models.BooleanField()

