import os

from latex_decompiler.cfg import StraightlineCFG
from latex_decompiler.train import (
    train_fixed_motifs,
    train_latex_e2e,
    train_single_digit_motifs,
)
from pixel_art.data.single_character_dataset import load_stamps

PIXEL_ART_RIGHT_ABOVE_LINE = 5.02262129e-05


def cfg(stamps_spec):
    return StraightlineCFG(sorted(load_stamps(stamps_spec)))


def motif_model_architecture(stamps_spec):
    return dict(
        type="ConvolutionalMotifModelMultipleSizes",
        in_channels=1,
        out_channels=64,
        sizes_per_character=1,
        # dummy for compatibility
        dataset_configuration=dict(min_dpi=100, max_dpi=200),
        cfg=cfg(stamps_spec),
        cr=8,
    )


class Experiment:
    def __init__(self, full_path, train_seed):
        self.path = ".".join(os.path.basename(full_path).split(".")[:-1])
        self.train_seed = int(train_seed)

    @property
    def model_checkpoint_path(self):
        if hasattr(self, "_model_checkpoint_path_override"):
            return self._model_checkpoint_path_override
        return f"model/{self.path}_{self.train_seed}"


class E2EExperiment(Experiment):
    def __init__(self, full_path, train_seed, stamps_spec):
        super().__init__(full_path, train_seed)
        self.data_spec = dict(
            type="PixelArtDomainDataset",
            domain_spec=dict(
                type="StampCircleDomain",
                size=100,
                min_radius=20,
                random_shift=3,
                max_syms=6,
                pre_noise=True,
                post_noise=0.05,
            ),
            stamps_spec=stamps_spec,
        )
        self.architecture = dict(
            type="LaTeXPredictor",
            channels=512,
            encoder_spec=dict(
                type="RowLSTMTransformerEncoder",
                row_lstm_spec=dict(type="BasicLSTM", bidirectional=True),
                transformer_encoder_spec=dict(
                    type="TransformerEncoder", nhead=8, layers=6
                ),
            ),
            decoder_spec=dict(
                type="TransformerCFGDecoder",
                cfg=cfg(stamps_spec),
                transformer_decoder_spec=dict(
                    type="TransformerDecoder", nhead=8, layers=6
                ),
            ),
            motifs_spec=motif_model_architecture(stamps_spec),
            sparsity_spec=dict(
                type="EnforceSparsityUniversally",
                starting_sparsity=0.5,
            ),
            post_sparse_spec=dict(
                type="CollapseMotifs",
                num_motifs=len(load_stamps(stamps_spec)),
                size_reduction=8,
            ),
        )
        self.total_steps = 6 * 10**6
        self.batch_size = 10
        self.lr = 1e-5
        self.suo_spec = dict(
            type="LinearThresholdAdaptiveSUO",
            initial_threshold=1,
            minimal_threshold=0,
            maximal_threshold=1,
            threshold_decrease_per_iter=1e-7,
            minimal_update_frequency=0,
            information_multiplier=0.75,
        )

        self.validation_spec = dict(type="TokenValidation")

        self.val_percent = 0.5

        self.motif_loss_specs = []

        self.done_at_density = 4.492e-05

        self.val_every = 20_000

        self.device = "cuda"

    def run(self):
        train_latex_e2e(
            path=self.model_checkpoint_path,
            data_spec=self.data_spec,
            architecture=self.architecture,
            train_seed=self.train_seed,
            val_seed=-1,
            total_steps=self.total_steps,
            batch_size=self.batch_size,
            print_every=1000,
            val_every=self.val_every,
            val_percent=self.val_percent,
            lr=self.lr,
            suo_spec=self.suo_spec,
            motif_loss_specs=self.motif_loss_specs,
            done_at_density=self.done_at_density,
            validation_spec=self.validation_spec,
            device=self.device,
        )

    def set_num_motifs(self, num_motifs, prev_num_motifs=None):
        prev_channels = (
            prev_num_motifs or self.architecture["sparsity_spec"]["channels"]
        )
        self.architecture["sparsity_spec"]["channels"] = num_motifs
        self.architecture["motifs_spec"]["num_motifs"] = self.architecture[
            "sparsity_spec"
        ]["channels"]
        self.architecture["post_sparse_spec"]["num_motifs"] = self.architecture[
            "sparsity_spec"
        ]["channels"]
        self.done_at_density *= (
            prev_channels / self.architecture["sparsity_spec"]["channels"]
        )

    def latex_architecture(self, cfg, *, num_motifs):
        self.architecture["sparsity_spec"] = dict(
            type="SparseLayerWithBatchNorm",
            underlying_sparsity_spec=dict(
                type="EnforceSparsityPerChannel2D",
                enforce_sparsity_per_channel_spec=dict(
                    type="EnforceSparsityPerChannelAccumulated",
                    accumulation_stop_strategy=dict(
                        type="StopAtFixedNumberMotifs", num_motifs=10
                    ),
                ),
            ),
            affine=False,
            starting_sparsity=0.5,
            channels=10,
        )
        self.architecture["motifs_spec"] = dict(
            type="ConvolutionalMotifModelMultipleSizes",
            in_channels=1,
            out_channels=64,
            sizes_per_character=1,
            # dummy for compatibility
            dataset_configuration=self.architecture["motifs_spec"][
                "dataset_configuration"
            ],
            cfg=self.architecture["motifs_spec"]["cfg"],
            cr=32,
        )
        self.architecture["decoder_spec"]["cfg"] = cfg

        self.set_num_motifs(num_motifs, 10)


class SingleDigitMotifsExperiment(Experiment):
    def __init__(self, full_path, train_seed):
        super().__init__(full_path, train_seed)

        self.architecture = None
        self.data_spec = None

        self.total_steps = 10**5
        self.batch_size = 10
        self.print_every = 1000
        self.val_every = 10_000

        self.lr = 1e-5

    def run(self):
        assert self.architecture is not None
        assert self.data_spec is not None
        train_single_digit_motifs(
            path=self.model_checkpoint_path,
            architecture=self.architecture,
            data_spec=self.data_spec,
            train_seed=self.train_seed,
            val_seed=-1,
            total_steps=self.total_steps,
            batch_size=self.batch_size,
            print_every=self.print_every,
            val_every=self.val_every,
            val_callback=lambda *args, **kwargs: {},
            val_percent=0.1,
            lr=self.lr,
        )


class FixedMotifsTrainingExperiment(Experiment):
    def __init__(self, full_path, train_seed):
        super().__init__(full_path, train_seed)

        self.architecture = None
        self.data_spec = None

        self.total_steps = 10**5
        self.batch_size = 10
        self.print_every = 1000
        self.val_every = 10_000

        self.lr = 1e-5

    def run(self):
        assert self.architecture is not None
        assert self.data_spec is not None
        train_fixed_motifs(
            path=self.model_checkpoint_path,
            architecture=self.architecture,
            data_spec=self.data_spec,
            train_seed=self.train_seed,
            val_seed=-1,
            total_steps=self.total_steps,
            batch_size=self.batch_size,
            print_every=self.print_every,
            val_every=self.val_every,
            val_callback=lambda *args, **kwargs: {},
            val_percent=0.1,
            lr=self.lr,
        )
