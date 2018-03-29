from deeplabv3plus import deeplabv3_plus_xception_light
from utils import load_data,dice_coef
import os
import pandas as pd
from keras.callbacks import CSVLogger,ModelCheckpoint,EarlyStopping
from matplotlib import pyplot as plt
import  numpy as np
from keras.optimizers import SGD
def train(train_rate,weights_dir,batch_size,pretrained_model=None,mode2="simple",mode3="color",train_dir="../data/stage1_train"):
  train_imgs,val_imgs,train_masks,val_masks,train_files,val_files,train_sizes,val_sizes=load_data(train_dir,train_rate,mode2=mode2,mode3=mode3)
  print(np.max(train_imgs))
  print(np.max(train_masks))
  model=deeplabv3_plus_xception_light(input_shape=(128,128,3),out_stride=16,num_classes=1)
  model.compile(
                optimizer=SGD(lr=3e-4, momentum=0.95),
#                optimizer=SGD(),
#                optimizer="adam",
                loss=["binary_crossentropy"],
                metrics=[dice_coef]
                )
  model.summary()
  print("Get Deeplabv3+!")
  
  

  if pretrained_model!=None:
    model.load_weights(pretrained_model)
    print("Loading model from "+pretrained_model)
     
  if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
  for i in range(5):
    fig=plt.figure()
    fig.add_subplot(121)
    plt.imshow(train_imgs[i])
    fig.add_subplot(122)
    plt.imshow(train_masks[i][:,:,0],cmap=plt.get_cmap('gray_r'))
    plt.show()
    
  model.fit(train_imgs,train_masks,
            epochs=500,
            batch_size=batch_size,
            validation_data=(val_imgs,val_masks),
            verbose=1,
            callbacks=[
                      CSVLogger(weights_dir+"training_log.csv"),
                      ModelCheckpoint(weights_dir+"last_checkpoint.hdf5",
                                                monitor='val_dice_coef', mode='max', save_best_only=True, 
                                                save_weights_only=False, verbose=0),
                      EarlyStopping(monitor="val_dice_coef",mode="max",patience=100)
                                ]
            )  

  df=pd.read_csv(weights_dir+"training_log.csv")
  df.to_csv(weights_dir+"training_log.csv",index=False)
weight_path="deeplabv3plus_xception_light_simple_color"
train(0.9,"weights/"+weight_path+"/",32)