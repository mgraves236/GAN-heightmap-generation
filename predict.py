import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam

optimizer = Adam(0.0002, 0.5) # learning rate and momentum
generator = tf.keras.models.load_model('model/generator_model_35000.h5')
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

#To create random images each time...
vector = np.random.randn(500) #Vector of random numbers (creates a column, need to reshape)
vector = vector.reshape(1, 500)

# generate image
X = generator.predict(vector)

# plot the result
plt.imshow(X[0, :, :, 0], cmap='gray_r')
plt.show()
