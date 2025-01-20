from keras.models import load_model
import tensorflow as tf
from keras import backend as K
import os
import cv2
import numpy as np
from colorama import Fore
import time
from preprocessing import crop_img


class TumobrainorModel:
    def __init__(self, model_path):
        # loading model from path
        with K.get_session().graph.as_default():
            self.model = load_model(model_path)
            self.model._make_predict_function()
            self.graph = tf.compat.v1.get_default_graph()

    def predict_data(self, data):
        # making predictions with loaded model
        with self.graph.as_default():
            output = self.model.predict(data)
            return output

# measuring time
start_time = time.time()
# creating model instance
model = TumobrainorModel("models/model-16-0.98-0.10.h5")

for filename in os.listdir("testing_images"):
    img = cv2.imread(os.path.join("testing_images", filename))
    # applying some transformations and normalization
    img = cv2.bilateralFilter(img, 2, 50, 50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = crop_img(img)
    img = cv2.resize(img, (200, 200))
    img = img / 255.

    # making prediction
    output = model.predict_data(np.expand_dims(img, axis=0))

    print(Fore.CYAN + str(np.argmax(output)), end=" ")
    print(Fore.RED + filename)

end_time = time.time()
print(Fore.GREEN + str(end_time - start_time), "seconds")
