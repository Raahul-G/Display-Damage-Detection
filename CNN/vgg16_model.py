import tensorflow as tf
from keras.applications import VGG16
from keras import layers, Model
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import pandas as pd

train_data_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\Train'
valid_data_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\Dataset\Valid'

batch_size = 16
image_size = (224, 224)

train_dataset = image_dataset_from_directory(
    train_data_dir,
    labels="inferred",
    label_mode="binary",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    seed=42,
)

valid_dataset = image_dataset_from_directory(
    valid_data_dir,
    labels="inferred",
    label_mode="binary",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False,
    seed=42,
)

base_model = VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
flatten_layer = layers.Flatten()(base_model.output)
dense_layer = layers.Dense(128, activation="relu")(flatten_layer)
output_layer = layers.Dense(1, activation="sigmoid")(dense_layer)

# Create the new model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy",  tf.keras.metrics.Precision() , tf.keras.metrics.Recall()]
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10,restore_best_weights = True)
mc = tf.keras.callbacks.ModelCheckpoint('model_best.h5', save_best_only=True)

class CustomCallBack(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('accuracy')>0.99):
                print("\nReached 99.0% accuracy so cancelling training!")
                self.model.stop_training = True

mycallback = CustomCallBack()

# Train the model
epochs = 10
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    callbacks= [early_stopping_cb,mc,mycallback]
)

# Convert history to a pandas DataFrame
history_df = pd.DataFrame(history.history)

# Save the history DataFrame as a CSV file
history_df.to_csv("training_history.csv", index=False)

save_dir = r'C:\FV_2.0\Projects\Display-Damage-Detection\CNN'

# Plot training & validation loss values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{save_dir}/loss_plot.png')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'{save_dir}/acc_plot.png')

plt.tight_layout()
plt.show()