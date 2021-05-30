# preparatory - DCGAN

## DCGAN

A generator network maps vectors of shape \(latent\_dim,\) to images of shape \(32, 32, 3\). A discriminator network maps images of shape \(32, 32, 3\) to a binary score estimating the probability that the image is real. A gan network chains the generator and the discriminator together: gan\(x\) = discriminator\(generator\(x\)\). Thus this gan network maps latent space vectors to the discriminator's assessment of the realism of these latent vectors as decoded by the generator. We train the discriminator using examples of real and fake images along with "real"/"fake" labels, as we would train any regular image classification model. To train the generator, we use the gradients of the generator's weights with regard to the loss of the gan model. This means that, at every step, we move the weights of the generator in a direction that will make the discriminator more likely to classify as "real" the images decoded by the generator. I.e. we train the generator to fool the discriminator.

```text
import keras
keras.__version__
```

First, develop a generator model, which turns a vector \(from the latent space -- during training it will sampled at random\) into a candidate image. One of the many issues that commonly arise with GANs is that the generator gets stuck with generated images that look like noise. A possible solution is to use dropout on both the discriminator and generator.

```text
import keras
from keras import layers
import numpy as np

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

# First, transform the input into a 16x16 128-channels feature map
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

# Then, add a convolution layer
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Upsample to 32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

# Few more conv layers
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Produce a 32x32 1-channel feature map
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()
```

Then, develop a discriminator model, that takes as input a candidate image \(real or synthetic\) and classifies it into one of two classes, either "generated image" or "real image that comes from the training set".

```text
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# One dropout layer - important trick!
x = layers.Dropout(0.4)(x)

# Classification layer
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

# To stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer.
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
```

Finally, setup the GAN, which chains the generator and the discriminator. This is the model that, when trained, will move the generator in a direction that improves its ability to fool the discriminator. This model turns latent space points into a classification decision, "fake" or "real", and it is meant to be trained with labels that are always "these are real images". So training gan will updates the weights of generator in a way that makes discriminator more likely to predict "real" when looking at fake images. Very importantly, set the discriminator to be frozen during training \(non-trainable\): its weights will not be updated when training gan. If the discriminator weights could be updated during this process, then we would be training the discriminator to always predict "real", which is not what we want!

```text
# Set discriminator weights to non-trainable
# (will only apply to the `gan` model)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
```

Load custom dataset

```text
#load and use file
import numpy as np

path = "ws_dataset.npz"
with np.load(path) as data:
    load ws_dataset as ws_data
    ws_data = data['ws_dataset']
```

Take a look at the data

```text
ws_data.shape
```

Attempt to flow from directory, adapted from Chollet's 'Fine tuning a Keras model'

```text
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'
# dimensions of input images.
img_width, img_height = 150, 150
```

```text
train_data_dir = 'ws_train'
validation_data_dir = 'ws_test'
nb_train_samples = 46396
nb_validation_samples = 18558
epochs = 50
batch_size = 16
```

```text
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
```

```text
test_datagen = ImageDataGenerator(rescale=1. / 255)
```

```text
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')
```

```text
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')
```

```text
# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)
```

Train the DCGAN

 Set up the training loop for each epoch: \* Draw random points in the latent space \(random noise\). \* Generate images with `generator` using this random noise. \* Mix the generated images with real ones. \* Train `discriminator` using these mixed images, with corresponding targets, either "real" \(for the real images\) or "fake" \(for the generated images\). \* Draw new random points in the latent space. \* Train `gan` using these random vectors, with targets that all say "these are real images". This will update the weights of the generator \(only, since discriminator is frozen inside `gan`\) to move them towards getting the discriminator to predict "these are real images" for generated images, i.e. this trains the generator to fool the discriminator.

```text
import os
from keras.preprocessing import image
import numpy as np
```

```text

# Load ws_data
(x_train, y_train), (_, _) = ws_data

path = "ws_dataset.npz"

with np.load(path) as ws_data:
    Xtrain_data = data['ws_data_X']
    print(train_data)
    test_data = data['ws_data_Y']
    print(test_data)

# Select images
x_train = x_train[y_train.flatten() == 6]
```

```text
x_train = ws_data

# Normalize data
x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
```

```text
# Normalize data (Chollet)
x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.
```

```text

iterations = 10000
batch_size = 20
save_dir = '/ws_gan'

# Start training loop
start = 0
for step in range(iterations):
    # Sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Decode them to fake images
    generated_images = generator.predict(random_latent_vectors)

    # Combine them with real images
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    # Assemble labels discriminating real from fake images
    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    # Add random noise to the labels - important trick!
    labels += 0.05 * np.random.random(labels.shape)

    # Train the discriminator
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Assemble labels that say "all real images"
    misleading_targets = np.zeros((batch_size, 1))

    # Train the generator (via the gan model,
    # where the discriminator weights are frozen)
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
      start = 0

    # Occasionally save / plot
    if step % 100 == 0:
        # Save model weights
        gan.save_weights('gan.h5')

        # Print metrics
        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        # Save one generated image
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_ws' + str(step) + '.png'))

        # Save one real image, for comparison
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_wa' + str(step) + '.png'))
```

Display a few fake images

```text
import matplotlib.pyplot as plt

# Sample random points in the latent space
random_latent_vectors = np.random.normal(size=(10, latent_dim))

# Decode them to fake images
generated_images = generator.predict(random_latent_vectors)

for i in range(generated_images.shape[0]):
    img = image.array_to_img(generated_images[i] * 255., scale=False)
    plt.figure()
    plt.imshow(img)
    
plt.show()
```

