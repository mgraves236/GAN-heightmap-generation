import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.ops import disable_eager_execution
from keras.datasets import mnist
from PIL import Image
import os
from os import listdir
from keras.utils import image_utils

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
disable_eager_execution()
img_rows = 253
img_cols = 253
channels = 1
img_shape = (img_rows, img_cols, channels)

noise_s = 500


def load_images():
    images = []
    for filename in os.listdir('/content/drive/MyDrive/PTG/real2'):
        img_path = os.path.join('/content/drive/MyDrive/PTG/real2', filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale (1 channel)
        img_data = np.array(img)
        images.append(img_data)

    images = np.array(images)
    images = (images.astype(np.float32) - 127.5) / 127.5
    images = np.expand_dims(images, axis=3)

    return images


X_train = load_images()

print(X_train.shape)

def build_generator():

  # random input vector
  noise_shape = (noise_s,)
  # three dense layers, between each batch normalization
  model = Sequential()
  model.add(Dense(256, input_shape=noise_shape))
  # allow a small gradient when the unit is not active
  model.add(LeakyReLU(alpha=0.2))
  # momentum -- how fast it's trained
  model.add(BatchNormalization(momentum=0.8))
  model.add(Dense(512))
  model.add(LeakyReLU(alpha=0.2))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Dense(1024))
  model.add(LeakyReLU(alpha=0.2))
  model.add(BatchNormalization(momentum=0.8))

  model.add(Dense(np.prod(img_shape), activation='tanh'))
  model.add(Reshape(img_shape))

  # model.summary()

  noise = Input(shape=noise_shape)
  # generated image
  img = model(noise)

  return Model(noise, img)


# score -- probability of the image being real or fake
def build_discriminator():

  model = Sequential()
  # three dense layers
  model.add(Flatten(input_shape=img_shape))
  model.add(Dense(512))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dense(256))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dense(1, activation='sigmoid'))

  # model.summary()

  img = Input(shape=img_shape)
  score = model(img)

  return Model(img, score)


def train(epochs=10000, batch_size=128, save_interval=200):

  half_batch = int(batch_size / 2)

  print ("Epoch \t Discriminator loss: \t accuracy: \t Generator loss: \t \n")
  for epoch in range(epochs):
    # -----------------------------------------
    # Train discriminator
    # -----------------------------------------

    # low, high, size return random integers
    index = np.random.randint(0, X_train.shape[0], half_batch)
    images = X_train[index]

    noise = np.random.normal(0, 1, (half_batch, noise_s))

    # # generate another half of batch of fake images
    generated_imgs = generator.predict(noise)

    # train, Y value -- labels, 0 for fake, 1 for real
    d_loss_real = discriminator.train_on_batch(images, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_imgs, np.zeros((half_batch, 1)))

    # average loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # -----------------------------------------
    # Train generator
    # -----------------------------------------

    noise = np.random.normal(0, 1, (batch_size, noise_s))

    # generate array in column format, all fake images set to real to fool the
    # discriminator
    valid_y = np.array([1] * batch_size)

    g_loss = combined.train_on_batch(noise, valid_y)

    # print ("Epoch \t Discriminator loss: \t accuracy: \t Generator loss: \t \n",  epoch, "\t", d_loss[0], "\t", 100*d_loss[1], "\t", g_loss)
    print (epoch, "\t", d_loss[0], "\t", 100*d_loss[1], "\t", g_loss)
    # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    if epoch % save_interval == 0:
      save_imgs(epoch)
      generator.save('/content/drive/MyDrive/PTG/model4/generator_model_' + str(epoch) + '.h5')
      discriminator.save('/content/drive/MyDrive/PTG/model4/discriminator_model_' + str(epoch) + '.h5')
    if epoch == epochs - 1:
      save_imgs(epoch)
      generator.save('/content/drive/MyDrive/PTG/model4/generator_model_' + str(epoch) + '.h5')
      discriminator.save('/content/drive/MyDrive/PTG/model4/discriminator_model_' + str(epoch) + '.h5')


def save_imgs(epoch):
    # # generate n random images
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, noise_s))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[count, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            count += 1
    fig.savefig('/content/drive/MyDrive/PTG/generated4/' + str(epoch) + '.png')
    plt.close()

#
# optimizer = Adam(0.0002, 0.5) # learning rate and momentum
#
# # trian discriminator
# discriminator = build_discriminator()
# discriminator.compile(loss='binary_crossentropy',
#                       optimizer = optimizer,
#                       metrics=['accuracy'])
#
# generator = build_generator()
# generator.compile(loss='binary_crossentropy', optimizer=optimizer)
#
# # random noise input
# z = Input(shape=(noise_s,))
# img = generator(z)
#
# # in a combined model discriminator is not being trained
# discriminator.trainable = False
#
# valid = discriminator(img)
#
# combined = Model(z, valid)
# combined.compile(loss='binary_crossentropy', optimizer=optimizer)
#
# train(epochs=100000, batch_size=256, save_interval=250)


# -------------------------------------------------------------------
# -------------------------------------------------------------------
optimizer = Adam(0.0002, 0.5) # learning rate and momentum

discriminator =  tf.keras.models.load_model('/content/drive/MyDrive/PTG/model4/10k/discriminator_model_9999.h5')
discriminator.compile(loss='binary_crossentropy',
                      optimizer = optimizer,
                      metrics=['accuracy'])

generator = tf.keras.models.load_model('/content/drive/MyDrive/PTG/model4/10k/generator_model_9999.h5')

generator.compile(loss='binary_crossentropy', optimizer=optimizer)

# random noise input
z = Input(shape=(noise_s,))
img = generator(z)

# in a combined model discriminator is not being trained
discriminator.trainable = False

valid = discriminator(img)

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

train(epochs=50000, batch_size=256, save_interval=200)