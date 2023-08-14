import tensorflow as tf
import numpy as np
import os
from keras.utils import load_img, img_to_array
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt

positive_pairs_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\positive_pairs'
negative_pairs_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\\negative_pairs'

K.clear_session()

image_pairs = []
labels = []

for pair_dir in os.listdir(positive_pairs_dir):

    img1 = img_to_array(load_img(os.path.join(positive_pairs_dir, pair_dir, "image1.jpg"))) / 255.0
    img2 = img_to_array(load_img(os.path.join(positive_pairs_dir, pair_dir, "image2.jpg"))) / 255.0
    label_path = os.path.join(positive_pairs_dir, pair_dir, "label.txt")


    with open(label_path, "r") as f:
        label = int(f.read())

    image_pairs.append((img1, img2))
    labels.append(label)
for pair_dir in os.listdir(negative_pairs_dir):

    img1 = img_to_array(load_img(os.path.join(positive_pairs_dir, pair_dir, "image1.jpg"))) / 255.0
    img2 = img_to_array(load_img(os.path.join(positive_pairs_dir, pair_dir, "image2.jpg"))) / 255.0
    label_path = os.path.join(positive_pairs_dir, pair_dir, "label.txt")

    with open(label_path, "r") as f:
        label = int(f.read())

    image_pairs.append((img1, img2))
    labels.append(label)

image_pairs = np.array(image_pairs)
labels = np.array(labels).reshape((len(labels), 1))


left_input = []
right_input = []

for pair in image_pairs:
    left = np.squeeze(pair[0:1, :, :, :])
    right = np.squeeze(pair[1:2, :, :, :])

    left_input.append(left)
    right_input.append(right)

left_input = np.array(left_input)
right_input = np.array(right_input)

def siamese_generator(left_input, right_input, labels, batch_size):
    num_samples = left_input.shape[0]
    while True:
        indices = np.random.randint(0, num_samples, batch_size)
        batch_left = left_input[indices]
        batch_right = right_input[indices]
        label = labels[indices]
        yield [batch_left, batch_right], label

batch_size = 12

dataset = tf.data.Dataset.from_generator(
    generator=lambda: siamese_generator(left_input, right_input, labels, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tf.float64, name='left_input'),
        tf.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tf.float64, name='right_input'),
        tf.TensorSpec(shape=(batch_size, 1), dtype=tf.int64, name='label_output'),
    )
)

generator = siamese_generator(left_input, right_input, labels, batch_size)

num_batches = int(np.ceil(len(left_input) / batch_size))

def build_siamese_model(input_shape):
    # Shared subnetwork
    input_left = Input(shape=input_shape, name="left_input")
    input_right = Input(shape=input_shape, name="right_input")

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='Conv_1')
    flatten = Flatten(name='Flat_1')
    dense1 = Dense(128, activation='relu', name='Dense_1')

    encoded_left = dense1(flatten(conv1(input_left)))
    encoded_right = dense1(flatten(conv1(input_right)))

    # Merge the two encoded outputs
    merged_vector = tf.keras.layers.Subtract()([encoded_left, encoded_right])
    prediction = Dense(1, activation='sigmoid', name='Output')(merged_vector)

    siamese_model = Model(inputs=[input_left, input_right], outputs=prediction)

    return siamese_model

input_shape = (224, 224, 3)

model = build_siamese_model(input_shape)
# model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision() , tf.keras.metrics.Recall()])

class CustomCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.99):
            print("\nReached 99.0% accuracy so cancelling training!")
            self.model.stop_training = True

mycallback = CustomCallBack()

model_history = model.fit(generator,
                          steps_per_epoch=num_batches,
                          epochs=20,
                          callbacks= [mycallback])

model.save_weights("model.h5")

# def plot_history(history):
#     accuracy = history.history['accuracy']
#     loss = history.history['loss']
#
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(accuracy, label='Training Accuracy')
#     plt.title('Training  Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(loss, label='Training Loss')
#     plt.title('Training Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid()
#
#     plt.tight_layout()
#     plt.show()

# plot_history(model_history)
