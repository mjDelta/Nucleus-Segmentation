from keras.layers import Input
from keras.applications.mobilenet import MobileNet
from keras.models import Model
MOBILENET_PATH="mobilenet_1_0_224_tf_no_top.h5"
class BaseBackend(object):
	def __init__(self,input_size):
		raise NotImplementedError("error")
	def normalize(self,img):
		raise NotImplementedError("error")
	def get_output_shape(self):
		return self.feature_extractor.get_output_shape_at(-1)[1:3]
	def extract(self,img):
		return self.feature_extractor(img)

class MobilenetFeature(BaseBackend):
	def __init__(self,input_size):
		input_=Input(shape=(input_size,input_size,3))
		
		mobilenet=MobileNet(input_shape=(224,224,3),include_top=False)
		mobilenet.load_weights(MOBILENET_PATH)
		
		x=mobilenet(input_)
		self.feature_extractor=Model(input_,x)
		self.feature_extractor.summary()
	def normalize(self,img):
		img/=255.
		img-=0.5
		img*=2
		return img
if __name__=="__main__":
	net=MobilenetFeature(128)
	