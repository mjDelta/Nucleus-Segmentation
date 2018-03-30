from keras.layers import Conv2D,concatenate,add,UpSampling2D,Input,BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def tiny_block(filters,input_,s=(1,1)):
	x=Conv2D(filters,(3,3),strides=s,padding="same",use_bias=False)(input_)
	x=BatchNormalization()(x)
	x=LeakyReLU(alpha=0.1)(x)
	return x	
def extract_block(filters,input_):
	res=Conv2D(filters,(1,1),strides=(2,2),padding="same",use_bias=False)(input_)
	res=BatchNormalization()(res)
	res=LeakyReLU(alpha=0.1)(res)

	skip=tiny_block(filters,input_)
	x=tiny_block(filters,skip,s=(2,2))
	x=add([res,x])	
	return x,skip
def expand_block(filters,input_):
	x=tiny_block(filters,input_)
	x=tiny_block(filters,x)
	x=UpSampling2D()(x)
	return x
def res_unet(input_size,filters=8):
	input_img=Input(shape=(input_size,input_size,3))

	##Extracting Path
	x1,skip1=extract_block(filters,input_img)
	x2,skip2=extract_block(filters*2,x1)
	x3,skip3=extract_block(filters*4,x2)
	x4,skip4=extract_block(filters*8,x3)

	temp=expand_block(filters*16,x4)

	##Expanding Path
	con1=concatenate([temp,skip4])
	ux1=expand_block(filters*8,con1)

	con2=concatenate([ux1,skip3])
	ux2=expand_block(filters*4,con2)

	con3=concatenate([ux2,skip2])
	ux3=expand_block(filters*2,con3)

	con4=concatenate([ux3,skip1])
	ux4=expand_block(filters,con4)

	out=Conv2D(1,(1,1),padding="same",use_bias=False)(ux4)
	model=Model(input_img,out)
	model.summary()
	return model
if __name__=="__main__":
	model=res_unet(128)





