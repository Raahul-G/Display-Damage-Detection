import tensorflow as tf
import numpy as np
import os
from keras.models import Model
from keras.utils import Sequence, load_img, img_to_array
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras import backend as K

positive_pairs_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\positive_pairs'
negative_pairs_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\negative_pairs'

def initialize_base_network():  # to build a base network with VGG16 architecture
    VGG = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3,))
    for layer in VGG.layers:
        layer.trainable = True

    last_layer = VGG.get_layer('block5_pool')
    print('last layer output shape ', last_layer.output_shape)
    last_output = last_layer.output

    x = Flatten(name="flatten_input")(last_output)
    x = Dense(512, activation='relu', name="first_base_dense")(x)
    x = Dropout(0.1, name="first_dropout")(x)
    x = Dense(256, activation='relu', name="second_base_dense")(x)
    x = Dropout(0.1, name="second_dropout")(x)
    output = Dense(128, activation='relu', name="third_base_dense")(x)
    model1 = tf.keras.models.Model(inputs=VGG.input, outputs=output)

    return model1

base_network = initialize_base_network()

# create the left input and point to the base network
input_a = Input(shape=(224, 224, 3,), name="left_input")
vect_output_a = base_network(input_a)

# create the right input and point to the base network
input_b = Input(shape=(224, 224, 3,), name="right_input")
vect_output_b = base_network(input_b)

# measure the similarity of the two vector outputs
L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([vect_output_a, vect_output_b])

# Add a dense layer with a sigmoid unit to generate the similarity score
output = tf.keras.layers.Dense(1, activation='sigmoid')(L1_distance)

model = Model([input_a, input_b], output)

# List image pairs and their corresponding labels
image_pairs = []
labels = []

# Load image pairs and labels from directories
for pair_dir in os.listdir(positive_pairs_dir):
    img1 = load_img(os.path.join(positive_pairs_dir, pair_dir, "image1.jpg"))
    img2 = load_img(os.path.join(positive_pairs_dir, pair_dir, "image2.jpg"))
    label_path = os.path.join(positive_pairs_dir, pair_dir, "label.txt")

    with open(label_path, "r") as f:
        label = int(f.read())

    image_pairs.append((img1, img2))
    labels.append(label)

for pair_dir in os.listdir(negative_pairs_dir):
    img1 = load_img(os.path.join(negative_pairs_dir, pair_dir, "image1.jpg"))
    img2 = load_img(os.path.join(negative_pairs_dir, pair_dir, "image2.jpg"))
    label_path = os.path.join(negative_pairs_dir, pair_dir, "label.txt")

    with open(label_path, "r") as f:
        label = int(f.read())

    image_pairs.append((img1, img2))
    labels.append(label)

# Convert images and labels to arrays
image_pairs = np.array(image_pairs)
labels = np.array(labels)

# Define a custom generator to generate batches of pairs and labels
class SiameseGenerator(Sequence):
    def __init__(self, image_pairs, labels, batch_size, image_size):
        self.image_pairs = image_pairs
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.indexes = np.arange(len(self.image_pairs))

    def __len__(self):
        return int(np.ceil(len(self.image_pairs) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indices = self.indexes[start_idx:end_idx]

        batch_pairs = self.image_pairs[batch_indices]
        batch_labels = self.labels[batch_indices]

        batch_images1 = np.array([img_to_array(img) / 255.0 for img in batch_pairs[:, 0]])
        batch_images2 = np.array([img_to_array(img) / 255.0 for img in batch_pairs[:, 1]])

        return [batch_images1, batch_images2], batch_labels


# Instantiate the custom generator
batch_size = 16
image_size = (224, 224)

# Compile the Siamese model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision() , tf.keras.metrics.Recall()])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10,restore_best_weights = True)
mc =tf.keras.callbacks.ModelCheckpoint('BaselineSiamese.h5', save_best_only=True)

class CustomCallBack(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('accuracy')>0.99):
                print("\nReached 99.0% accuracy so cancelling training!")
                self.model.stop_training = True

mycallback = CustomCallBack()

# Train the Siamese model using the generator
model_history = model.fit(SiameseGenerator, epochs=10, callbacks= [early_stopping_cb,mc,mycallback])

