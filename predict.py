import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import cv2

optimizer = Adam(0.0002, 0.5)  # learning rate and momentum
generator = tf.keras.models.load_model('model/generator_model_35000.h5')
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

vector = np.random.randn(500)
vector = vector.reshape(1, 500)
# generate image
X = generator.predict(vector)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
dst = cv2.filter2D(X[0, :, :, 0],-1,kernel)
# plot the result
plt.imshow(dst, cmap='gray_r')
plt.axis('off')
plt.savefig('/content/drive/MyDrive/PTG/exp.png',bbox_inches='tight')
plt.show()
plt.close()
