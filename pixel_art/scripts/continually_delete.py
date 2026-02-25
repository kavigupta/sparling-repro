import datetime
import time

from latex_decompiler.cleanup import remove_steps_from_all
from pixel_art.analysis.audio_mnist_experiment import models as audio_models
from pixel_art.analysis.latex_experiment import models as latex_models

while True:
    print(datetime.datetime.now())
    print("LaTeX models")
    remove_steps_from_all(
        latex_models,
        min_window=10**10,
        max_density_to_keep=float("inf"),
        function_to_load=lambda path, step: None,
        loop=False,
    )
    print("Audio models")
    remove_steps_from_all(
        audio_models,
        min_window=10**10,
        max_density_to_keep=float("inf"),
        function_to_load=lambda path, step: None,
        loop=False,
    )
    print("Waiting")

    time.sleep(60 * 60)
