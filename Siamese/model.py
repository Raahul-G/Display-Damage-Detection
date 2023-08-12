import tensorflow as tf
from keras.utils import img_to_array, load_img
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
import os

# Define paths to your positive_pairs and negative_pairs directories
positive_pairs_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\positive_pairs'
negative_pairs_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\negative_pairs'

# Function to encode image and label into a tf.Example
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a TFRecord writer
tfrecord_filename = "siamese.tfrecords"
writer = tf.io.TFRecordWriter(tfrecord_filename)

# Load and process image pairs and labels
for pair_dir in os.listdir(positive_pairs_dir):
    img1 = img_to_array(load_img(os.path.join(positive_pairs_dir, pair_dir, "image1.jpg")))
    img2 = img_to_array(load_img(os.path.join(positive_pairs_dir, pair_dir, "image2.jpg")))
    label_path = os.path.join(positive_pairs_dir, pair_dir, "label.txt")

    with open(label_path, "r") as f:
        label = int(f.read())

    # Serialize images and label into a tf.Example
    feature = {
        'image1': _bytes_feature(img1.tobytes()),
        'image2': _bytes_feature(img2.tobytes()),
        'label': _int64_feature(label)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Write the serialized example to the TFRecord file
    writer.write(example.SerializeToString())

for pair_dir in os.listdir(negative_pairs_dir):
    img1 = img_to_array(load_img(os.path.join(negative_pairs_dir, pair_dir, "image1.jpg")))
    img2 = img_to_array(load_img(os.path.join(negative_pairs_dir, pair_dir, "image2.jpg")))
    label_path = os.path.join(negative_pairs_dir, pair_dir, "label.txt")

    with open(label_path, "r") as f:
        label = int(f.read())

    # Serialize images and label into a tf.Example
    feature = {
        'image1': _bytes_feature(img1.tobytes()),
        'image2': _bytes_feature(img2.tobytes()),
        'label': _int64_feature(label)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Write the serialized example to the TFRecord file
    writer.write(example.SerializeToString())

# Close the TFRecord writer
writer.close()

# Function to parse the serialized examples from the TFRecords
def parse_example(example):
    feature_description = {
        'image1': tf.io.FixedLenFeature([], tf.string),
        'image2': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)

    image1 = tf.io.decode_raw(parsed_example['image1'], tf.uint8)
    image2 = tf.io.decode_raw(parsed_example['image2'], tf.uint8)
    label = parsed_example['label']

    print("Decoded Image 1 shape:", image1.shape)
    print("Decoded Image 2 shape:", image2.shape)

    # Reshape and normalize images
    image1 = tf.reshape(image1, (224, 224, 3))
    image1 = tf.cast(image1, tf.float32) / 255.0

    image2 = tf.reshape(image2, (224, 224, 3))
    image2 = tf.cast(image2, tf.float32) / 255.0

    print("Decoded Image 1 shape after reshape :", image1.shape)
    print("Decoded Image 2 shape after reshape :", image2.shape)

    return {'left_input': image1, 'right_input': image2}, label

# Create a TFRecord dataset
tfrecord_filename = r'C:\FV_2.0\Projects\Display-Damage-Detection\Siamese\siamese.tfrecords'
dataset = tf.data.TFRecordDataset(tfrecord_filename)
dataset = dataset.map(parse_example)

# Define the base network
def initialize_base_network():
    VGG = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    for layer in VGG.layers:
        layer.trainable = False

    last_layer = VGG.get_layer('block5_pool')
    last_output = last_layer.output

    x = Flatten(name="flatten_input")(last_output)
    x = Dense(512, activation='relu', name="first_base_dense")(x)
    x = Dropout(0.1, name="first_dropout")(x)
    x = Dense(256, activation='relu', name="second_base_dense")(x)
    x = Dropout(0.1, name="second_dropout")(x)
    output = Dense(128, activation='relu', name="third_base_dense")(x)
    model = Model(inputs=VGG.input, outputs=output)

    return model

base_network = initialize_base_network()


# Define the Siamese model
input_shape = (224, 224, 3)  # Define the input shape consistent with your base network
input_a = Input(shape=input_shape, name="left_input")
input_b = Input(shape=input_shape, name="right_input")

vect_output_a = base_network(input_a)
vect_output_b = base_network(input_b)

L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([vect_output_a, vect_output_b])

output = Dense(1, activation='sigmoid', name='OutputLayer')(L1_distance)

siamese_model = Model(inputs=[input_a, input_b], outputs=output)

# Compile the Siamese model
siamese_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Instantiate the custom generator
batch_size = 16  # Adjust batch size as needed

print(siamese_model.summary())

# Train the Siamese model using the generator
model_history = siamese_model.fit(dataset.batch(batch_size), epochs=10)