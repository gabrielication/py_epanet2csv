import tensorflow as tf
# import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import layers

leak_area = "0164" # "0246" #"0164"
leak_group = 7
leak_group_model = 7

dataset_path = "tensorflow_group_datasets/model/h5/"
model_filename = "model_leak_group"+str(leak_group)+"_train_node_"+str(leak_group_model)+".h5"
loaded_model = tf.keras.models.load_model(dataset_path+model_filename)

output_filename = "tensorflow_group_datasets/fig/model_simple.png"

tf.keras.utils.plot_model(loaded_model, to_file=output_filename, show_shapes=True)

print(output_filename + " saved.")