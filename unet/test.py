from keras.optimizers import SGD
import pandas as pd
import numpy as np
from deeplabv3plus import deeplabv3_plus_xception_light
from utils import load_data,dice_coef,draw_contours,prob_to_rles
from skimage.transform import resize
import math
def test(train_rate,weights_dir,mode2="smart",mode3="simple",test_dir="../data/stage1_test"):
  temp_px=128
  model=deeplabv3_plus_xception_light(input_shape=(128,128,3),out_stride=16,num_classes=1)
  model.compile(
                optimizer=SGD(lr=2e-5,momentum=0.99),
#                optimizer="adam",
                loss=["binary_crossentropy"],
                metrics=[dice_coef])
  model.load_weights(weights_dir)
  
  test_imgs,test_files,test_sizes=load_data(test_dir,train_rate,"test",mode2=mode2,mode3=mode3)

  test_preds=model.predict(test_imgs)>0.5
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
	weight_path="deeplabv3plus_xception_light_samrt_color"
	test(0.95,"weights/"+weight_path+"/last_checkpoint.hdf5")