# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:44:59 2018

@author: ZMJ
"""
import keras.backend as K
from skimage.io import imread
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist as ea
from skimage.transform import resize
from skimage.morphology import label
import time
import tensorflow as tf
from keras.preprocessing import image 
import math
from skimage.measure import find_contours
import scipy
from tqdm import tqdm
import pandas as pd
from dual_IDG import DualImageDataGenerator 
def data_generator(X_train,Y_train,mode,batch_size,seed=5):
  if mode=="train":
    gen=DualImageDataGenerator(
                               horizontal_flip=True, vertical_flip=True,
                               rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                               zoom_range=(0.9, 1.1),
                               fill_mode='constant', cval=0.0
                               )
  else:
    gen=DualImageDataGenerator()
  return gen.flow(X_train,Y_train,batch_size=batch_size)
#  # Creating the training Image and Mask generator
#  image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
#  mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
#  
#  # Keep the same seed for image and mask generators so they fit together
#  
#  image_datagen.fit(X_train[:int(X_train.shape[0]*0.9)], augment=True, seed=seed)
#  mask_datagen.fit(Y_train[:int(Y_train.shape[0]*0.9)], augment=True, seed=seed)
#  
#  x=image_datagen.flow(X_train[:int(X_train.shape[0]*0.9)],batch_size=batch_size,shuffle=False, seed=seed)
#  y=mask_datagen.flow(Y_train[:int(Y_train.shape[0]*0.9)],batch_size=batch_size,shuffle=False, seed=seed)
#  
#  
#  
#  # Creating the validation Image and Mask generator
#  image_datagen_val = image.ImageDataGenerator()
#  mask_datagen_val = image.ImageDataGenerator()
#  
#  image_datagen_val.fit(X_train[int(X_train.shape[0]*0.9):], augment=True, seed=seed)
#  mask_datagen_val.fit(Y_train[int(Y_train.shape[0]*0.9):], augment=True, seed=seed)
#  
#  x_val=image_datagen_val.flow(X_train[int(X_train.shape[0]*0.9):],batch_size=batch_size,shuffle=False, seed=seed)
#  y_val=mask_datagen_val.flow(Y_train[int(Y_train.shape[0]*0.9):],batch_size=batch_size,shuffle=False, seed=seed)
#  return x,y,x_val,y_val
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def mean_IOU(X,Y):
  ##compute mean iou of X(batch_size,width,height)
#  print(X.shape)
#  X_clip=K.clip(K.batch_flatten(X),0.,1.)
#  Y_clip=K.clip(K.batch_flatten(Y),0.,1.)
#  X_clip=K.cast(K.greater(X_clip,0.5),"float32")
#  Y_clip=K.cast(K.greater(Y_clip,0.5),"float32")
#  
#  intersection=K.sum(X_clip*Y_clip,axis=1)
#  union=K.sum(K.maximum(X_clip,Y_clip),axis=1)
#  union=K.switch(K.equal(union,0),K.ones_like(union),union)
#  return K.mean(intersection/K.cast(union,"float32"))
    """Computes mean Intersection-over-Union (IOU) for two arrays of binary images.
    Assuming X and Y are of shape (n_images, w, h)."""
    X=tf.cast(X,tf.float32)
    Y=tf.cast(Y,tf.float32)
    X_fl = K.clip(K.batch_flatten(X), 0., 1.)
    Y_fl = K.clip(K.batch_flatten(Y), 0., 1.)
    X_fl = K.greater(X_fl, 0.5)
    Y_fl = K.greater(Y_fl, 0.5)
    X_fl=tf.cast(X_fl,tf.float32)
    Y_fl=tf.cast(Y_fl,tf.float32)
    intersection = K.sum(X_fl * Y_fl, axis=1)
    union = K.sum(K.maximum(X_fl, Y_fl), axis=1)  
    union=tf.cond(tf.equal(tf.reduce_sum(union),0),lambda: tf.ones(tf.shape(union)),lambda: union)	
    return K.mean(intersection / K.cast(union, 'float32'))
#    prec = []
#    for t in np.arange(0.5, 1.0, 0.05):
#        y_pred_ = tf.to_int32(Y > t)
#        score, up_opt = tf.metrics.mean_iou(X, y_pred_, 2)
#        K.get_session().run(tf.local_variables_initializer())
#        with tf.control_dependencies([up_opt]):
#            score = tf.identity(score)
#        prec.append(score)
#    return K.mean(K.stack(prec), axis=0)
def dice(y_true,y_pred):
#  X_clip=K.clip(K.batch_flatten(X),0,1)
#  Y_clip=K.clip(K.batch_flatten(Y),0,1)
#
#  intersection=2*K.sum(X_clip*Y_clip,axis=1)
#  union=K.sum(X_clip*X_clip,axis=1)+K.sum(Y_clip*Y_clip,axis=1)
#  union=K.switch(K.equal(union,0),K.ones_like(union),union)
#
#  return K.mean(intersection/union)
    y_true=tf.cast(y_true,tf.float32)
    y_pred=tf.cast(y_pred,tf.float32)
    y_true_f = K.clip(K.batch_flatten(y_true), 0., 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), 0., 1.)

    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)

    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    union=tf.cond(tf.equal(tf.reduce_sum(union),0),lambda: tf.ones(tf.shape(union)),lambda: union)

    return K.mean(intersection / union)
def dice_metrics(X,Y):
  X_f=K.cast(K.greater(X,0.5),"float32")
  Y_f=K.cast(K.greater(Y,0.5),"float32")
  return dice(X_f,Y_f)
def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def load_img_from_dir(dir_,count=1000,mode="train",mode2="simple",mode3="color",temp_px=128):
  print(temp_px)
  if mode2=="simple":
    print("Loading data method:resize simple.")
    imgs=[]
    file_names=[]
    masks=[]
    sizes=[]
    ##缩放到128px实验
    for i,f in enumerate(os.listdir(dir_)):
      
      img=imread(os.path.join(dir_,f,"images",f+".png"))/255.
      img=ea(img)
      if mode3=="gray":
        img=rgb2gray(img)
        img=np.expand_dims(img,axis=-1)
      imgs.append(resize(img,(temp_px,temp_px),mode='constant', preserve_range=True))##add equalize_adapthist 
      file_names.append(f)
      sizes.append((img.shape[0],img.shape[1]))
  
      if mode=="test":
        continue
      mask=np.zeros((imgs[0].shape[0],imgs[0].shape[1],1))
      for mask_f in os.listdir(os.path.join(dir_,f,"masks")):
        m=resize(imread(os.path.join(dir_,f,"masks",mask_f)),(temp_px,temp_px),mode='constant', preserve_range=True)>128.
        m=np.expand_dims(m,axis=-1)
        mask+=m
      masks.append(mask)
      if i==count:
        break
    if mode=="test":
      return np.array(imgs),"",np.array(file_names),np.array(sizes)
    return np.array(imgs),np.array(masks),np.array(file_names),np.array(sizes)
    return imgs,masks,file_names,sizes
  elif mode2=="smart":
    print("Loading data method:cut smart.")

    imgs=[]
    file_names=[]
    masks=[]
    sizes=[]
    ##裁剪到128px实验
    for i,f in enumerate(os.listdir(dir_)):
      if i%50==0:
        print(i,end="\t")
      img=imread(os.path.join(dir_,f,"images",f+".png"))/255.
#      print(np.max(img))
      img=ea(img)
#      plt.imshow(img)
#      plt.show()
      if mode3=="gray":
        img=rgb2gray(img)        
        img=np.expand_dims(img,axis=-1)
#      print(img.shape)
      new_img,cut_num_h,cut_num_w=fill_image(img,temp_px)
      for row in range(cut_num_h):
        for col in range(cut_num_w):
          imgs.append(new_img[row*temp_px:(row+1)*temp_px,col*temp_px:(col+1)*temp_px,:])      
          file_names.append(f)
          sizes.append((img.shape[0],img.shape[1]))
  
      if mode=="test":
        continue
      mask=np.zeros((cut_num_h*temp_px,cut_num_w*temp_px,1))
      for mask_f in os.listdir(os.path.join(dir_,f,"masks")):
        m=imread(os.path.join(dir_,f,"masks",mask_f))>128.
        m=np.expand_dims(m,axis=-1)
        new_m,_,_=fill_image(m,temp_px) 
        mask+=new_m
      for row in range(cut_num_h):
        for col in range(cut_num_w):
          masks.append(mask[row*temp_px:(row+1)*temp_px,col*temp_px:(col+1)*temp_px,:])      
      if i==count:
        break
    print()
    if mode=="test":
      return np.array(imgs),"",np.array(file_names),np.array(sizes)
    return np.array(imgs),np.array(masks),np.array(file_names),np.array(sizes) 
def fill_image(img,temp_px):
  img_h,img_w,img_ch=img.shape
#  print("Shape:"+str(img.shape))
  cut_num_h=math.ceil(float(img_h)/temp_px);cut_num_w=math.ceil(float(img_w)/temp_px)
      
  new_img_h=cut_num_h*temp_px;new_img_w=cut_num_w*temp_px
  new_img=np.zeros((new_img_h,new_img_w,img_ch))
  if new_img_h*new_img_w>img_h*img_w:
    for row in range(cut_num_h):
      for column in range(cut_num_w):
        flip_w=new_img_w-img_w
        flip_h=new_img_h-img_h        
        if row!=cut_num_h-1 and column!=cut_num_w-1:
          new_img[row*temp_px:(row+1)*temp_px,column*temp_px:(column+1)*temp_px,:]=img[row*temp_px:(row+1)*temp_px,column*temp_px:(column+1)*temp_px,:]
        elif row!=cut_num_h-1 and column==cut_num_w-1:
         
          new_img[row*temp_px:(row+1)*temp_px,column*temp_px:img_w,:]=img[row*temp_px:(row+1)*temp_px,column*temp_px:,:]
          if flip_w>0:
            temp_img=h_flip(img[row*temp_px:(row+1)*temp_px,-flip_w:,:])
            new_img[row*temp_px:(row+1)*temp_px,img_w:,:]=temp_img
        elif row==cut_num_h-1 and column!=cut_num_w-1:

          new_img[row*temp_px:img_h,column*temp_px:(column+1)*temp_px,:]=img[row*temp_px:,column*temp_px:(column+1)*temp_px,:]
          if flip_h>0:
            temp_img=v_flip(img[-flip_h:,column*temp_px:(column+1)*temp_px,:])          
            new_img[img_h:,column*temp_px:(column+1)*temp_px,:]=temp_img
        elif row==cut_num_h-1 and column==cut_num_w-1:

          new_img[row*temp_px:img_h,column*temp_px:img_w,:]=img[row*temp_px:,column*temp_px:,:]
          if flip_w>0:
            temp_img=h_flip(img[row*temp_px:,-flip_w:,:])
            new_img[row*temp_px:img_h,img_w:,:]=temp_img
          if flip_h>0:
            temp_img2=v_flip(img[-flip_h:,column*temp_px:,:])
            new_img[img_h:,column*temp_px:img_w,:]=temp_img2
          if flip_h>0 and flip_w>0:
            temp_img3=v_flip(h_flip(img[-flip_h:,-flip_w:,:]))
            new_img[img_h:,img_w:,:]=temp_img3
  else:
    new_img[:,:,:]=img
  return new_img,cut_num_h,cut_num_w
def h_flip(img):
  return np.fliplr(img)
def v_flip(img):
  return np.flipud(img)
def log_dice_loss(X,Y):
  return -K.log(dice(X,Y)+1e-5)

def minus_dice_loss(X,Y):
  return 1.-dice(X,Y)
def quarter_minus_dice_loss(X,Y):
  return (1.-dice(X,Y))*0.25
def a_minus_dice_loss(X,Y):
  return quarter_minus_dice_loss(X,Y)*0.5
def b_minus_dice_loss(X,Y):
  return quarter_minus_dice_loss(X,Y)*0.5
def c_minus_dice_loss(X,Y):
  return quarter_minus_dice_loss(X,Y)*1
def d_minus_dice_loss(X,Y):
  return quarter_minus_dice_loss(X,Y)*2
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def result_visulization(imgs,masks,ys,sizes):
  idx=np.random.choice(len(imgs),1)
  print(imgs[idx].shape)
  fig=plt.figure(figsize=(12,4))
  fig.add_subplot(131)
  plt.imshow(np.squeeze(imgs[idx]))
  fig.add_subplot(132)
  plt.imshow(np.squeeze(masks[idx]))
  fig.add_subplot(133)
  plt.imshow(np.squeeze(ys[idx])>0)
  plt.show() 
  
def load_data(dir_,train_rate,mode="train",mode2="simple",mode3="color",temp_px=128):
  start=time.time()
  imgs,masks,files,sizes=load_img_from_dir(dir_,mode=mode,mode2=mode2,mode3=mode3) 
#  masks=np.expand_dims(masks,axis=-1)
  if mode=="test":
    print("Loading Testing Data Cost Time:"+str(time.time()-start)+"s")
    print("Testing img num:"+str(len(imgs)))
    if temp_px!=128:
      resize_imgs=[]
      for img in imgs:
        resize_imgs.append(resize(img,(temp_px,temp_px),mode='constant', preserve_range=True))
      resize_imgs=np.array(resize_imgs)
      print(resize_imgs.shape)
      return resize_imgs
    return imgs,files,sizes

  all_idxs=np.arange(0,len(imgs))
  np.random.shuffle(all_idxs)
  train_idxs=all_idxs[:int(train_rate*len(imgs))]
  val_idxs=all_idxs[int(train_rate*len(imgs)):]
  train_imgs=imgs[train_idxs]
  val_imgs=imgs[val_idxs]
  if temp_px!=128:
    resize_imgs_train=[]
    for img in train_imgs:
      resize_imgs_train.append(resize(img,(temp_px,temp_px),mode='constant', preserve_range=True))
    resize_imgs_test=[]
    for img in val_imgs:
      resize_imgs_test.append(resize(img,(temp_px,temp_px),mode='constant', preserve_range=True))
    print("Loading Training Data Cost Time:"+str(time.time()-start)+"s")
    print("Training img num:"+str(len(train_imgs))+".\t Validation img num:"+str(len(val_imgs)))
    resize_imgs_train=np.array(resize_imgs_train)
    resize_imgs_test=np.array(resize_imgs_test)
    print(resize_imgs_train.shape)
    return resize_imgs_train,resize_imgs_test
  train_masks=masks[train_idxs]
  val_masks=masks[val_idxs]
  train_files=files[train_idxs]
  val_files=files[val_idxs]
  train_sizes=sizes[train_idxs]
  val_sizes=sizes[val_idxs]
  print("Loading Training Data Cost Time:"+str(time.time()-start)+"s")
  print("Training img num:"+str(len(train_imgs))+".\t Validation img num:"+str(len(val_imgs)))
  print(train_imgs.shape)
  return train_imgs,val_imgs,train_masks,val_masks,train_files,val_files,train_sizes,val_sizes
def draw_contours(img,pred,mask,filename,weight_name):
  fig=plt.figure(figsize=(8,8))
  ax=fig.add_subplot(1,1,1)
  ax.imshow(img)
  dir_="seg_test_vis/"+weight_name+"/"
  if not os.path.exists(dir_):
    os.makedirs(dir_)
  contours1=find_contours(pred,0.5)
  for contour in contours1:
    ax.plot(contour[:,1],contour[:,0],linewidth=0.5,color="red")##preds
  if mask!=None:
    contours2=find_contours(mask,0.5)
    for contour in contours2:
      ax.plot(contour[:1],contour[:,0],linewidth=0.5,color="blue")##mask
  plt.savefig(dir_+filename+".png")
  plt.show() 
  
##smart sub path:的彩色图片分割效果不佳 ，将其彩色分割替换成simple_sub_path的提交
def merge_smart_simple(smart_sub_path,simple_sub_path):
  smart_df=pd.read_csv(smart_sub_path)
  simple_df=pd.read_csv(simple_sub_path)
  color_sub=pd.read_csv("../data/stage1_test_colorful_image.csv",header=None)
  colors=[str(color_sub.iloc[i,0]) for i in range(len(color_sub))]
  out_df=smart_df.copy()
  simple_df_need=pd.DataFrame()
  smart_drop_idxs=[]
  for i in range(len(colors)):
    smart_drop_idxs.extend(list(out_df[out_df["ImageId"]==colors[i]].index))
    simple_df_need=pd.concat([simple_df_need,simple_df[simple_df["ImageId"]==colors[i]]])
  out_df=out_df.drop(smart_drop_idxs)
  out_df=pd.concat([out_df,simple_df_need])
  out_df.to_csv("subs/"+smart_sub_path[5:-4]+simple_sub_path[5:-4]+".csv",index=False)
#  return smart_df,simple_df,out_df,smart_drop_idxs
if __name__=="__main__":
##  import warnings
##  warnings.filterwarnings("ignore")
#  train_dir="../data/train"
##  save_path="../data/stage1_train_arange/"
##  if not os.path.exists(save_path+"imgs/"):
##    os.makedirs(save_path+"imgs/")
##  if not os.path.exists(save_path+"masks/"):
##    os.makedirs(save_path+"masks/")
#  imgs,masks,files,sizes=load_img_from_dir(train_dir,mode="train",count=10,mode2="smart",mode3="color")
##  for i in tqdm(range(len(imgs))):
##    img=imgs[i]
##    mask=masks[i]
##    file=files[i]
##    img_path=save_path+"imgs/"+file+".png"
##    mask_path=save_path+"masks/"+file+".png"
##    scipy.misc.toimage(img*255, cmin=0, cmax=255).save(img_path)
##    scipy.misc.toimage(mask, cmin=0, cmax=255).save(mask_path)
##  x,y,x_val,y_val=data_generator(imgs,masks,"train",1)
#  gen=data_generator(imgs,masks,"train",3)
#  xs=[];ys=[]
#  for i in range(6):
#    img,mask=gen.next()
#    xs.append(img[0]);ys.append(mask[0])
#    xs.append(img[1]);ys.append(mask[1])
#    xs.append(img[2]);ys.append(mask[2])
#  img=imgs[0,:,:,:]
##  mask=masks[0,:,:,0]
#  for i in range(2):
#    fig=plt.figure(figsize=(8,12))
#    fig.add_subplot(231)
#    plt.imshow(np.squeeze(xs[i*3+0]), cmap = plt.get_cmap('gray'))
#    fig.add_subplot(232)    
#    plt.imshow(np.squeeze(xs[i*3+1]), cmap = plt.get_cmap('gray'))
#    fig.add_subplot(233)     
#    plt.imshow(np.squeeze(xs[i*3+2]), cmap = plt.get_cmap('gray'))
##    fig.add_subplot(224)     
##    plt.imshow(np.squeeze(y_val.next()[0].astype(np.uint8)), cmap = plt.get_cmap('gray'))
#    plt.show()
#  for i in range(2):
#    fig=plt.figure(figsize=(8,12))
#    fig.add_subplot(231)
#    plt.imshow(np.squeeze(ys[i*3+0]), cmap = plt.get_cmap('gray'))
#    fig.add_subplot(232)    
#    plt.imshow(np.squeeze(ys[i*3+1]), cmap = plt.get_cmap('gray'))
#    fig.add_subplot(233)     
#    plt.imshow(np.squeeze(ys[i*3+2]), cmap = plt.get_cmap('gray'))
##    fig.add_subplot(224)     
##    plt.imshow(np.squeeze(y_val.next()[0].astype(np.uint8)), cmap = plt.get_cmap('gray'))
#    plt.show()
#  fig=plt.figure(figsize=(9,6))
#  for i in range(2):
#    
#    fig.add_subplot(2,3,i*3+1)
#    plt.imshow(imgs[i*3+0,:,:,:])
#    fig.add_subplot(2,3,i*3+2)    
#    plt.imshow(imgs[i*3+1,:,:,:])
#    fig.add_subplot(2,3,i*3+3)     
#    plt.imshow(imgs[i*3+2,:,:,:])
#  fig=plt.figure(figsize=(9,6))
#  for i in range(2):
#    
#    fig.add_subplot(2,3,i*3+1)
#    plt.imshow(masks[i*3+0,:,:,0], cmap = plt.get_cmap('gray'))
#    fig.add_subplot(2,3,i*3+2)    
#    plt.imshow(masks[i*3+1,:,:,0], cmap = plt.get_cmap('gray'))
#    fig.add_subplot(2,3,i*3+3)     
#    plt.imshow(masks[i*3+2,:,:,0], cmap = plt.get_cmap('gray'))    
###
###    plt.show()
  smart_sub_path="subs/pretrained 2 smart-sub.csv"
  simple_sub_path="subs/weights A_gray 500epoch_patience100 mnet_advanced base01 adam minus_dice_loss mean_IOU dropout channel_0.5_smart_gray-sub.csv"
  merge_smart_simple(smart_sub_path,simple_sub_path)