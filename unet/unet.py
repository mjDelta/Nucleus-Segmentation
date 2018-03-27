# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 08:43:35 2018

@author: ZMJ
"""
from keras.layers import Conv2D,MaxPooling2D,Input,concatenate,UpSampling2D,Dropout,AtrousConvolution2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model 
from keras.optimizers import SGD
from keras.preprocessing import image
from utils import *
import numpy as np
from keras.callbacks import CSVLogger,ModelCheckpoint,EarlyStopping
import os
import pandas as pd
from skimage.transform import resize
import math
from matplotlib import pyplot as plt
from keras import backend as K
def unet_architecture(scale_channel,img_row=128,img_col=128):
#  scale_channel=0.5
  inputs=Input(shape=(img_row,img_col,3),dtype="float32")
  x=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(inputs)
  x=Dropout(0.3)(x)  
  x_con1=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=MaxPooling2D()(x_con1)
  
  x=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x_con2=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=MaxPooling2D()(x_con2)

  x=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x_con3=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=MaxPooling2D()(x_con3) 

  x=Conv2D(int(scale_channel*512),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x_con4=Conv2D(int(scale_channel*512),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=MaxPooling2D()(x_con4)

  x=Conv2D(int(scale_channel*1024),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*1024),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=UpSampling2D()(x)
  
  x=concatenate([x,x_con4])
  x=Conv2D(int(scale_channel*512),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*512),(3,3),activation="relu",padding="same")(x)
  x=UpSampling2D()(x)
  
  x=concatenate([x,x_con3])
  x=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=UpSampling2D()(x)  
  
  x=concatenate([x,x_con2])
  x=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=UpSampling2D()(x)
  
  x=concatenate([x,x_con1])
  x=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Conv2D(1,(1,1),activation="sigmoid",padding="same")(x)
  
  model=Model(input=inputs,output=x)  
  return model
def mnet_architecture(scale_channel,img_row=128,img_col=128,channel=1):

  block1_inputs=Input(shape=(img_row,img_col,channel),dtype="float32",name="input1")
  block1_conv1=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block1_inputs)
  block1_drop1=Dropout(0.3)(block1_conv1)  
  block1_conv2=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block1_drop1)
  block1_pool1=MaxPooling2D()(block1_conv2)
  
  block2_inputs=Input(shape=(img_row//2,img_col//2,channel),dtype="float32",name="input2")  
  block2_conv1=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block2_inputs)
  block2_concat1=concatenate([block1_pool1,block2_conv1])
  block2_conv2=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block2_concat1)
  block2_drop1=Dropout(0.3)(block2_conv2)  
  block2_conv3=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block2_drop1)
  block2_pool1=MaxPooling2D()(block2_conv3)
  
  block3_inputs=Input(shape=(img_row//4,img_col//4,channel),dtype="float32",name="input3")  
  block3_conv1=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block3_inputs)
  block3_concat1=concatenate([block2_pool1,block3_conv1])
  block3_conv2=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block3_concat1)
  block3_drop1=Dropout(0.3)(block3_conv2)  
  block3_conv3=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block3_drop1)
  block3_pool1=MaxPooling2D()(block3_conv3) 
  
  block4_inputs=Input(shape=(img_row//8,img_col//8,channel),dtype="float32",name="input4")  
  block4_conv1=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block4_inputs)
  block4_concat1=concatenate([block3_pool1,block4_conv1])
  block4_conv2=Conv2D(int(scale_channel*512),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block4_concat1)
  block4_drop1=Dropout(0.3)(block4_conv2)  
  block4_conv3=Conv2D(int(scale_channel*512),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block4_drop1)
  block4_pool1=MaxPooling2D()(block4_conv3)

  x=Conv2D(int(scale_channel*1024),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block4_pool1)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*1024),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=UpSampling2D()(x)
  
  x=concatenate([x,block4_conv3])
  x=Conv2D(int(scale_channel*512),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*512),(3,3),activation="relu",padding="same")(x)
  outputs2=UpSampling2D(size=(8,8))(x)
  outputs2=Conv2D(1,(1,1),activation="sigmoid",padding="same")(outputs2)
  x=UpSampling2D()(x)
  
  x=concatenate([x,block3_conv3])
  x=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  outputs3=UpSampling2D(size=(4,4))(x)
  outputs3=Conv2D(1,(1,1),activation="sigmoid",padding="same")(outputs3)
  x=UpSampling2D()(x)  
  
  x=concatenate([x,block2_conv3])
  x=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  outputs4=UpSampling2D()(x)
  outputs4=Conv2D(1,(1,1),activation="sigmoid",padding="same")(outputs4)
  x=UpSampling2D()(x)
  
  x=concatenate([x,block1_conv2])
  x=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  outputs5=Conv2D(1,(1,1),activation="sigmoid",padding="same")(x)
  x=concatenate([outputs2,outputs3,outputs4,outputs5])
  x=Conv2D(1,(1,1),activation="sigmoid",padding="same")(x)
  model=Model([block1_inputs,block2_inputs,block3_inputs,block4_inputs],x)  
#  model=Model([block1_inputs,block2_inputs,block3_inputs,block4_inputs],[outputs2,outputs3,outputs4,outputs5])  
  return model
def atrous_mnet_architecture(scale_channel,img_row=128,img_col=128,channel=1):

  block1_inputs=Input(shape=(img_row,img_col,channel),dtype="float32",name="input1")
  block1_conv1=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block1_inputs)
  block1_drop1=Dropout(0.3)(block1_conv1)  
  block1_conv2=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block1_drop1)
  block1_pool1=MaxPooling2D()(block1_conv2)
  
  block2_inputs=Input(shape=(img_row//2,img_col//2,channel),dtype="float32",name="input2")  
  block2_conv1=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block2_inputs)
  block2_concat1=concatenate([block1_pool1,block2_conv1])
  block2_conv2=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block2_concat1)
  block2_drop1=Dropout(0.3)(block2_conv2)  
  block2_conv3=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block2_drop1)
  block2_pool1=MaxPooling2D()(block2_conv3)
  
  block3_inputs=Input(shape=(img_row//4,img_col//4,channel),dtype="float32",name="input3")  
  block3_conv1=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block3_inputs)
  block3_concat1=concatenate([block2_pool1,block3_conv1])
  block3_conv2=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block3_concat1)
  block3_drop1=Dropout(0.3)(block3_conv2)  
  block3_conv3=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block3_drop1)
  block3_pool1=MaxPooling2D()(block3_conv3) 
  
  block4_conv1=Conv2D(int(scale_channel*128),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block3_pool1)
  block4_conv2=Conv2D(int(scale_channel*512),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block4_conv1)
  block4_drop1=Dropout(0.3)(block4_conv2)  
  block4_atrous1=AtrousConvolution2D(int(scale_channel*512),(3,3),atrous_rate=2,activation="relu",kernel_initializer='he_normal',border_mode="same")(block4_drop1)
  block4_conv3=Conv2D(int(scale_channel*512),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block4_atrous1)
  block4_drop2=Dropout(0.3)(block4_conv3)  
  block4_conv4=Conv2D(int(scale_channel*512),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(block4_drop2)
  x=UpSampling2D()(block4_conv4)

  x=concatenate([x,block3_conv3])
  x=Conv2D(int(scale_channel*512),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*512),(3,3),activation="relu",padding="same")(x)
  outputs2=UpSampling2D(size=(4,4))(x)
  outputs2=Conv2D(1,(1,1),activation="sigmoid",padding="same")(outputs2)
  x=UpSampling2D()(x)
  
  x=concatenate([x,block2_conv3])
  x=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*256),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  outputs3=UpSampling2D(size=(2,2))(x)
  outputs3=Conv2D(1,(1,1),activation="sigmoid",padding="same")(outputs3)
  x=UpSampling2D()(x)  
    
  x=concatenate([x,block1_conv2])
  x=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  x=Dropout(0.3)(x)  
  x=Conv2D(int(scale_channel*64),(3,3),activation="relu",kernel_initializer='he_normal',padding="same")(x)
  outputs5=Conv2D(1,(1,1),activation="sigmoid",padding="same")(x)
  x=concatenate([outputs2,outputs3,outputs5])
  x=Conv2D(1,(1,1),activation="sigmoid",padding="same")(x)
  model=Model([block1_inputs,block2_inputs,block3_inputs],x)  
#  model=Model([block1_inputs,block2_inputs,block3_inputs,block4_inputs],[outputs2,outputs3,outputs4,outputs5])  
  return model
def four_img_data_generator(imgs,masks,mode,batch_size,temp_px):
  while True:
    imgs_da2=[];imgs_da3=[];imgs_da4=[]
    train_gen=data_generator(imgs,masks,mode,batch_size)
    imgs_da,masks_da=train_gen.next()
    for img in imgs_da:
      imgs_da2.append(resize(img,(temp_px//2,temp_px//2),mode='constant', preserve_range=True))
      imgs_da3.append(resize(img,(temp_px//4,temp_px//4),mode='constant', preserve_range=True))
      imgs_da4.append(resize(img,(temp_px//8,temp_px//8),mode='constant', preserve_range=True))
#    yield [imgs_da,np.array(imgs_da2),np.array(imgs_da3),np.array(imgs_da4)],masks_da
    yield [imgs_da,np.array(imgs_da2),np.array(imgs_da3),np.array(imgs_da4)],[masks_da,masks_da,masks_da,masks_da]
def train(train_rate,weights_dir,batch_size,channels,pretrained_model=None,mode2="simple",mode3="color",train_dir="../data/stage1_train",channel=1):
  model=atrous_mnet_architecture(channels,channel=channel)
  model.compile(
#                optimizer=SGD(lr=2e-5),
#                optimizer=SGD(),
                optimizer="adam",
                loss=["binary_crossentropy"],
                metrics=[dice]
                )
  model.summary()
  print("Get M-Net!")
  
  
  train_imgs,val_imgs,train_masks,val_masks,train_files,val_files,train_sizes,val_sizes=load_data(train_dir,train_rate,mode2=mode2,mode3=mode3)
#  [imgs_da,imgs_da2,imgs_da3,imgs_da4],[masks_da,masks_da2,masks_da3,masks_da4]=four_img_data_generator(train_imgs,train_masks,"train",batch_size,128).__next__()
#  for i in range(4):
#    fig=plt.figure(figsize=(15,3))
#    fig.add_subplot(151)
#    plt.imshow(np.squeeze(imgs_da[i]), cmap = plt.get_cmap('gray'))
#    fig.add_subplot(152)
#    plt.imshow(np.squeeze(masks_da[i]), cmap = plt.get_cmap('gray'))
#    fig.add_subplot(153)
#    plt.imshow(np.squeeze(imgs_da2[i]), cmap = plt.get_cmap('gray'))
#    fig.add_subplot(154)
#    plt.imshow(np.squeeze(imgs_da3[i]), cmap = plt.get_cmap('gray'))
#    fig.add_subplot(155)
#    plt.imshow(np.squeeze(imgs_da4[i]), cmap = plt.get_cmap('gray'))
#    plt.show()    
  train_imgs2,val_imgs2=load_data(train_dir,train_rate,mode2=mode2,mode3=mode3,temp_px=128//2)
  train_imgs3,val_imgs3=load_data(train_dir,train_rate,mode2=mode2,mode3=mode3,temp_px=128//4)
#  train_imgs4,val_imgs4=load_data(train_dir,train_rate,mode2=mode2,mode3=mode3,temp_px=128//8)
  if pretrained_model!=None:
    model.load_weights(pretrained_model)
    print("Loading model from "+pretrained_model)
     
  if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

  model.fit({"input1":train_imgs,"input2":train_imgs2,"input3":train_imgs3},train_masks,
            epochs=500,
            batch_size=batch_size,
            validation_data=({"input1":val_imgs,"input2":val_imgs2,"input3":val_imgs3},val_masks),
            verbose=1,
            callbacks=[
                      CSVLogger(weights_dir+"training_log.csv"),
                      ModelCheckpoint(weights_dir+"last_checkpoint.hdf5",
                                                monitor='val_dice', mode='max', save_best_only=True, 
                                                save_weights_only=False, verbose=0),
                      EarlyStopping(monitor="val_dice",mode="max",patience=100)
                                ]
            )  
#  val_imgs2=[];val_imgs3=[];val_imgs4=[]
#  for val_img in val_imgs:
#    val_imgs2.append(resize(val_img,(128//2,128//2),mode='constant', preserve_range=True))
#    val_imgs3.append(resize(val_img,(128//4,128//4),mode='constant', preserve_range=True))
#    val_imgs4.append(resize(val_img,(128//8,128//8),mode='constant', preserve_range=True))
#  val_imgs2=np.array(val_imgs2)
#  val_imgs3=np.array(val_imgs3)
#  val_imgs4=np.array(val_imgs4)
#  model.fit_generator(four_img_data_generator(train_imgs,train_masks,"train",batch_size,128),
#                      epochs=50,
#                      steps_per_epoch=len(train_imgs)//batch_size,
#                      validation_data=([val_imgs,val_imgs2,val_imgs3,val_imgs4],[val_masks,val_masks,val_masks,val_masks]),
##                      validation_steps=len(val_imgs)//batch_size,
#                      verbose=1,
#                      callbacks=[
#                                CSVLogger(weights_dir+"training_log.csv"),
#                                ModelCheckpoint(weights_dir+"last_checkpoint.hdf5",
#                                                          monitor='val_loss', mode='min', save_best_only=True, 
#                                                          save_weights_only=False, verbose=0),
#                                EarlyStopping(monitor="val_loss",patience=5)
#                                          ]                      
#      )
  df=pd.read_csv(weights_dir+"training_log.csv")
  df.to_csv(weights_dir+"training_log.csv",index=False)
def test(train_rate,weights_dir,channels,mode2,mode3,channel=1,test_dir="../data/stage1_test"):
  temp_px=128
  model=mnet_architecture(channels,channel=channel)
  model.compile(
                optimizer=SGD(lr=2e-5,momentum=0.99),
#                optimizer="adam",
                loss=minus_dice_loss,
                metrics=[mean_IOU])
  model.load_weights(weights_dir)
  
  test_imgs,test_files,test_sizes=load_data(test_dir,train_rate,"test",mode2=mode2,mode3=mode3)
  test_imgs2=load_data(test_dir,train_rate,"test",mode2=mode2,mode3=mode3,temp_px=128//2)
  test_imgs3=load_data(test_dir,train_rate,"test",mode2=mode2,mode3=mode3,temp_px=128//4)
  test_imgs4=load_data(test_dir,train_rate,"test",mode2=mode2,mode3=mode3,temp_px=128//8)


  test_preds=model.predict([test_imgs,test_imgs2,test_imgs3,test_imgs4])>0.5
#  test_preds_=model.predict([test_imgs,test_imgs2,test_imgs3,test_imgs4])
#  test_preds=np.zeros((len(test_preds_[0]),128,128,1))
#  for i in range(len(test_preds_[0])):
#    for j in range(128):
#      for k in range(128):
#        test_preds[i,j,k]=np.mean([test_preds_[0][i,j,k],test_preds_[1][i,j,k],test_preds_[2][i,j,k],test_preds_[3][i,j,k]])
#  test_preds=test_preds>0.5
  preds_test_upsampled = []
  true_test_files=None
  if mode2=="simple":
#    result_visulization(train_imgs,train_masks,train_preds,train_sizes)
#    result_visulization(val_imgs,val_masks,val_preds,val_sizes)
    
    for i in range(len(test_preds)):
      pred_temp=resize(np.squeeze(test_preds[i]), 
                                         (test_sizes[i][0], test_sizes[i][1]), 
                                         mode='constant', preserve_range=True)
      preds_test_upsampled.append(pred_temp) 
      img_temp=resize(test_imgs[i], 
                                         (test_sizes[i][0], test_sizes[i][1]), 
                                         mode='constant', preserve_range=True)
      draw_contours(img_temp,pred_temp,None,test_files[i],weights_dir[8:-21])
    true_test_files=test_files
  elif mode2=="smart":
    last=""
    row=0;col=0;pred=None;img=None;w=0;h=0;w_num=0;h_num=0
    true_test_files=[]
    for i,test_f in enumerate(test_files):
      if test_f==last:
#        print("row :%s\tcol :%s"%(row,col))
        pred[row*temp_px:(row+1)*temp_px,col*temp_px:(col+1)*temp_px]=np.squeeze(test_preds[i])
        img[row*temp_px:(row+1)*temp_px,col*temp_px:(col+1)*temp_px,:]=test_imgs[i]
        if col<h_num-1:
          col+=1
        else:
          row+=1;col=0
        last=test_f        
      else:
        row=0;col=0;
        if i!=0:
          pred_temp=pred[:w,:h];img_temp=img[:w,:h,:]
          preds_test_upsampled.append(pred_temp)
          draw_contours(img_temp,pred_temp,None,last,weights_dir[8:-21])
#          scipy.misc.toimage(pred_temp*255, cmin=0, cmax=255).save("seg_test_mask/"+last+".png")

          true_test_files.append(last)
        w,h=test_sizes[i]
        w_num=math.ceil(float(w)/temp_px);h_num=math.ceil(float(h)/temp_px)
        print("w_num :%s\th_num :%s"%(w_num,h_num))
        pred=np.zeros((w_num*temp_px,h_num*temp_px))
        img=np.zeros((w_num*temp_px,h_num*temp_px,3))
        pred[row*temp_px:(row+1)*temp_px,col*temp_px:(col+1)*temp_px]=np.squeeze(test_preds[i])
        img[row*temp_px:(row+1)*temp_px,col*temp_px:(col+1)*temp_px,:]=test_imgs[i]
        if col<h_num-1:
          col+=1
        else:
          row+=1;col=0
        last=test_f
    pred_temp=pred[:w,:h];img_temp=img[:w,:h,:]
    preds_test_upsampled.append(pred_temp)
#    scipy.misc.toimage(pred_temp*255, cmin=0, cmax=255).save("seg_test_mask/"+last+".png")
    draw_contours(img_temp,pred_temp,None,last,weights_dir[8:-21])
    true_test_files.append(last)
  new_test_ids = []
  rles = []
  for n, id_ in enumerate(true_test_files):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
  sub = pd.DataFrame()
  sub['ImageId'] = new_test_ids
  sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
  sub.to_csv("subs/"+weights_dir[8:-21]+'-sub.csv', index=False)
#  return train_preds
if __name__=="__main__":
  import warnings
  warnings.filterwarnings("ignore")
  channels=0.5
  mode2="smart"
  mode3="gray"  
  import sys
  argv=sys.argv[1]
#  argv="test"
  batch_size=8
  if argv=="train":
    print("Training Begin.")
    weight_path="weights  500epoch_patience100 atrous_mnet_advanced base01 adam minus_dice_loss mean_IOU dropout channel_"+str(channels)+"_"+mode2+"_"+mode3
    train(0.8,"weights/"+weight_path+"/",batch_size,channels,mode2=mode2,mode3=mode3,train_dir="../data/stage1_train",channel=1)
  elif argv=="test":
#    weight_path="pretrained 2 "+mode2
    weight_path="weights A_gray 500epoch_patience100 mnet_advanced base01 adam minus_dice_loss mean_IOU dropout channel_0.5_smart_gray"
    print("Testing Begin.")
    test(0.8,"weights/"+weight_path+"/last_checkpoint.hdf5",channels,mode2=mode2,mode3=mode3,test_dir="../data/stage1_test_A_gray",channel=1)[0,:,:,0]
  elif argv=="pretrained":
    print("Pretrained training Begin.")
#    weight_path+="_epoch_100"
    weight_path="pretrained 2 "+mode2
    print("This model path:"+weight_path)
    loaded_model="weights/weights unet base01 sgd minus_dice_loss mean_IOU dropout channel_0.5/last_checkpoint.hdf5"
    train(0.8,"weights/"+weight_path+"/",batch_size,channels,pretrained_model=loaded_model,mode2=mode2)
    
