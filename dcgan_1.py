import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Dense, LeakyReLU, Input, Reshape, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
(train_images, _), (_, _) = mnist.load_data()
train_images = (train_images.astype(np.float32) - 127.5)/127.5
train_images = np.expand_dims(train_images, axis=3) # Add another axis for the channels since the default data is just a 2D matrix

BATCH_SIZE = 128
SAVE_INTERVAL = 50
EPOCHS = 1000


# The images' properties
img_rows = 28
img_cols = 28
channels = 1 # Adding the channels as their own axis in the data is good for flexibility
img_shape = (img_rows, img_cols, channels)

def build_generator():
    noise_shape = (100,)

    model = Sequential([
        Dense(256, input_shape=noise_shape), # Input is a random noise vector
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8), # Normalize after every Dense/LeakyRelU layer
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(1024),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(np.prod(img_shape), activation='tanh'), # Number of nodes correlates with the number of pixels in the original image
        Reshape(img_shape)
    ])

    noise = Input(shape=noise_shape) # Prepare the noise vector
    img = model(noise) # Feed the noise vector to the model

    return Model(noise, img) # Packs the noise vector and the result image into a single I/O Model

def build_discriminator():
    model = Sequential([
        Flatten(input_shape=img_shape),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation=tf.nn.sigmoid)
    ])
    '''model = Sequential([
        Conv2D(16, (2,2), activation='relu', input_shape=img_shape),
        MaxPool2D(),
        Conv2D(32, (2,2), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])'''

    img = Input(shape=img_shape) # Input layers takes the fake image made by the generator
    validity = model(img) # Prediction of the fake image's veracity

    return Model(img, validity) # Packs the image input and the [0,1] output into a single I/O Model

def train(epochs, batch_size=BATCH_SIZE, save_interval=SAVE_INTERVAL):
    half_batch = int(batch_size/2)
    for epoch in range(epochs):
        # Training the discriminator
        ids = np.random.randint(0, train_images.shape[0], half_batch) # Gets a random sample of indices from the dataset
        imgs = train_images[ids] # Fetches real images from the dataset with the random array indices

        noise = np.random.normal(0, 1, (half_batch, 100)) # Creates a bunch of noisy image data
        gen_imgs = generator.predict(noise) # Generates a half batch of fake images using noise

        # Train the discriminator separately
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

        d_loss = np.add(d_loss_real, d_loss_fake) # Sums the two losses (they should be as low as possible)

        # Training the generator
        noise = np.random.normal(0, 1, (batch_size, 100)) # Creating noise again for generator

        valid_y = np.array([1]*batch_size) # Creates an array of ones
        g_loss = combined.train_on_batch(noise, valid_y) # The outputs created by the noise will be compared against the intended output, 1 (which means that the discriminator thinks it's real)

        print(f'Epoch {epoch} || d_loss = {d_loss[0]} || acc = {d_loss[1]} || g_loss = {g_loss}')
        if epoch % save_interval == 0:
            save_images(epoch)

# Functions for saving the generated images
def save_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r*c, 100))
    gen_imgs = generator.predict(noise)

    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r,c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f'images/mnist_{epoch}.png')
    plt.close()

optimizer = tf.optimizers.Adam(0.0001)

discriminator = build_discriminator()
discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

generator = build_generator()
generator.compile(optimizer=optimizer, loss='binary_crossentropy')

z = Input(shape=(100,))
img = generator(z)
discriminator.trainable = False

valid = discriminator(img)

combined = Model(z, valid)
combined.compile(optimizer=optimizer, loss='binary_crossentropy')

train(epochs=5000, batch_size=32, save_interval=10)