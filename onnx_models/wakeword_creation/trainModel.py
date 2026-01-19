# @title  { display-mode: "form" }
# @markdown # 3. Train the Model
# @markdown Now that you have verified your target wake word and downloaded the data,
# @markdown the last step is to adjust the training paramaters (or keep
# @markdown the defaults below) and start the training!

# @markdown Each paramater controls a different aspect of training:
# @markdown - `number_of_examples` controls how many examples of your wakeword
# @markdown are generated. The default (1,000) usually produces a good model,
# @markdown but between 30,000 and 50,000 is often the best.

# @markdown - `number_of_training_steps` controls how long to train the model.
# @markdown Similar to the number of examples, the default (10,000) usually works well
# @markdown but training longer usually helps.

# @markdown - `false_activation_penalty` controls how strongly false activations
# @markdown are penalized during the training process. Higher values can make the model
# @markdown much less likely to activate when it shouldn't, but may also cause it
# @markdown to not activate when the wake word isn't spoken clearly and there is
# @markdown background noise.

# @markdown With the default values shown below,
# @markdown this takes about 30 - 60 minutes total on the normal CPU Colab runtime.
# @markdown If you want to train on more examples or train for longer,
# @markdown try changing the runtime type to a GPU to significantly speedup
# @markdown the example generating and model training.

# @markdown When the model finishes training, you can navigate to the `my_custom_model` folder
# @markdown in the file browser on the left (click on the folder icon), and download
# @markdown the [your target wake word].onnx or  <your target wake word>.tflite files.
# @markdown You can then use these as you would any other openWakeWord model!

# Load default YAML config file for training
import yaml
config = yaml.load(open("openwakeword/examples/custom_model.yml", 'r').read(), yaml.Loader)

# Modify values in the config and save a new version
number_of_examples = 1000 # @param {type:"slider", min:100, max:50000, step:50}
number_of_training_steps = 10000  # @param {type:"slider", min:0, max:50000, step:100}
false_activation_penalty = 1500  # @param {type:"slider", min:100, max:5000, step:50}
config["target_phrase"] = [target_word]
config["model_name"] = config["target_phrase"][0].replace(" ", "_")
config["n_samples"] = number_of_examples
config["n_samples_val"] = max(500, number_of_examples//10)
config["steps"] = number_of_training_steps
config["target_accuracy"] = 0.5
config["target_recall"] = 0.25
config["output_dir"] = "./my_custom_model"
config["max_negative_weight"] = false_activation_penalty

config["background_paths"] = ['./audioset_16k', './fma']  # multiple background datasets are supported
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}

with open('my_model.yaml', 'w') as file:
    documents = yaml.dump(config, file)

# Generate clips
!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --generate_clips

# Step 2: Augment the generated clips

!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips

# Step 3: Train model

!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model

# # Manually save to tflite as this doesn't work right in colab (broken in python 3.11, default in Colab as of January 2025)
# def convert_onnx_to_tflite(onnx_model_path, output_path):
#     """Converts an ONNX version of an openwakeword model to the Tensorflow tflite format."""
#     # imports
#     import onnx
#     import logging
#     import tempfile
#     from onnx_tf.backend import prepare
#     import tensorflow as tf

#     # Convert to tflite from onnx model
#     onnx_model = onnx.load(onnx_model_path)
#     tf_rep = prepare(onnx_model, device="CPU")
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tf_rep.export_graph(os.path.join(tmp_dir, "tf_model"))
#         converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(tmp_dir, "tf_model"))
#         tflite_model = converter.convert()

#         logging.info(f"####\nSaving tflite mode to '{output_path}'")
#         with open(output_path, 'wb') as f:
#             f.write(tflite_model)

#     return None

# convert_onnx_to_tflite(f"my_custom_model/{config['model_name']}.onnx", f"my_custom_model/{config['model_name']}.tflite")

# Convert ONNX model to tflite using `onnx2tf` library (works for python 3.11 as of January 2025)
onnx_model_path = f"my_custom_model/{config['model_name']}.onnx"
name1, name2 = f"my_custom_model/{config['model_name']}_float32.tflite", f"my_custom_model/{config['model_name']}.tflite"
!onnx2tf -i {onnx_model_path} -o my_custom_model/ -kat onnx____Flatten_0
!mv {name1} {name2}

# Automatically download the trained model files
from google.colab import files

files.download(f"my_custom_model/{config['model_name']}.onnx")
files.download(f"my_custom_model/{config['model_name']}.tflite")
