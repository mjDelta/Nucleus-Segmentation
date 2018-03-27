# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 17:18:07 2018

@author: ZMJ
"""
import os 
from matplotlib import pyplot as plt
from skimage.io import imread
import shutil

path="..\data\stage1_test"
if not os.path.exists(os.path.join("..\data","stage1_test_A_gray")):
  os.makedirs(os.path.join("..\data","stage1_test_A_gray"))
if not os.path.exists(os.path.join("..\data","stage1_test_B_color")):
  os.makedirs(os.path.join("..\data","stage1_test_B_color"))
dirs=["A_gray","B_color"]
A_gray_files=os.listdir(os.path.join("..\data","stage1_test_A_gray"))
B_color_files=os.listdir(os.path.join("..\data","stage1_test_B_color"))
for f in os.listdir(path):

  if f in A_gray_files:
    continue
  if f in B_color_files:
    continue
  print(os.path.join(path,f))
  plt.imshow(imread(os.path.join(path,f,"images",f+".png")))
  plt.show()
  print("0(黑白)    1(彩色)")
  label=input()
  while label=="" or (int(label)!=0 and int(label)!=1):
    print("0(黑白)    1(彩色)")
    label=input()    
  shutil.copytree(os.path.join(path,f),os.path.join("..\data","stage1_test_"+dirs[int(label)],f))


    