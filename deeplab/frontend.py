from keras.layers import Conv2D
from deeplabv3plus import aspp
	#aspp
	x=aspp(x,input_shape,out_stride,64)
	x=Conv2D(64,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	#x=Dropout(0.9)(x)
	
	##decoder 
	x=BilinearUpsampling((4,4))(x)
	dec_skip=Conv2D(48,(1,1),padding="same",use_bias=False)(skip)
	dec_skip=BatchNormalization()(dec_skip)
	dec_skip=Activation("relu")(dec_skip)
	x=Concatenate()([x,dec_skip])
	
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=Conv2D(32,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=Conv2D(32,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	
	x=Conv2D(num_classes,(1,1),padding="same",activation="sigmoid")(x)
	x=BilinearUpsampling((4,4))(x)
	model=Model(img_input,x)
	return model