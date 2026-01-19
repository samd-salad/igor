# @title  { display-mode: "form" }
# @markdown # 1. Test Example Training Clip Generation
# @markdown Since openWakeWord models are trained on synthetic examples of your
# @markdown target wake word, it's a good idea to make sure that the examples
# @markdown sound correct. Type in your target wake word below, and run the
# @markdown cell to listen to it.
# @markdown
# @markdown Here are some tips that can help get the wake word to sound right:

# @markdown - If your wake word isn't being pronounced in the way
# @markdown you want, try spelling out the sounds phonetically with underscores
# @markdown separating each part.
# @markdown For example: "hey siri" --> "hey_seer_e".

# @markdown - Spell out numbers ("2" --> "two")

# @markdown - Avoid all punctuation except for "?" and "!", and remove unicode characters

import os
import sys
from IPython.display import Audio
if not os.path.exists("./piper-sample-generator"):
    !cd piper-sample-generator && git checkout 213d4d5

    # Install system dependencies
    !pip install piper-tts piper-phonemize-cross
    !pip install webrtcvad
    !pip install 'torch<=2.5' torchvision torchaudio

    if "piper-sample-generator/" not in sys.path:
        sys.path.append("piper-sample-generator/")

    from generate_samples import generate_samples

target_word = 'hey' # @param {type:"string"}

def text_to_speech(text):
    generate_samples(text = text,
                max_samples=1,
                length_scales=[1.1],
                noise_scales=[0.7], noise_scale_ws = [0.7],
                output_dir = './', batch_size=1, auto_reduce_batch_size=True,
                file_names=["test_generation.wav"]
                )

text_to_speech(target_word)
Audio("test_generation.wav", autoplay=True)
