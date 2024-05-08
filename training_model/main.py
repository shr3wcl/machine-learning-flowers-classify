from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
img_size_224p = 128
model = load_model('./NhanDienHoaPBL5_40ep.h5') # ⚠️Can be Customized⚠️