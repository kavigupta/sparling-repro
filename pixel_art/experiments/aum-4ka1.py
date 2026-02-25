from sys import argv

from pixel_art.analysis.audio_mnist_experiment import (
    audio_mnist_data_spec_train,
    digits_dataset_spec,
)
from pixel_art.experiments.experiment import SingleDigitMotifsExperiment

exp = SingleDigitMotifsExperiment(__file__, argv[1])

N_MELS = 64

exp.architecture = dict(
    type="LaTeXPredictorJustMotifsModelStub",
    motifs_spec=dict(
        type="AudioConvolutionalMotifModel",
        in_channels=N_MELS,
        out_channels=64,
        cr=16,
        num_motifs=10,
    ),
    channels=64,
)

exp.data_spec = digits_dataset_spec(audio_mnist_data_spec_train)

exp.batch_size = 100

exp.lr = 1e-5

exp.val_every = 50_000
exp.total_steps = 3 * 10**6

exp.run()
