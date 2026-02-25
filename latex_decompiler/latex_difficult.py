# computed in https://github.com/asolarlez/ExpeditionsBioDev/blob/9b0973600e8a91b907da4f603c7adbdec9390f00/latex-domain/notebooks/latex-font.ipynb
from latex_decompiler.latex_cfg import LATEX_CFG_SPECS

FONT_GROUP_A = [
    "arev",
    "baskervaldx",
    "charterbt",
    "computer_modern_bright",
    "eb_garabond",
    "fira_sans_newtxsf",
    "gfs_neohellenic",
    "heuristica",
    "iwona",
    "iwona_condensed",
    "kerkis",
    "kp_sans",
    "lxfonts",
    "noto_serif",
    "step",
    "stickstoo",
    "utopia_fourier",
]

FONT_GROUP_B = [
    "antykwa_torunska",
    "antykwa_torunska_condensed",
    "antykwa_torunska_light",
    "antykwa_torunska_light_condensed",
    "baskervillef",
    "boisik",
    "computer_modern",
    "concmath",
    "concmath_euler",
    "kp_serif",
    "mlmodern",
    "new_px",
    "urw_schoolbook_l",
]


def difficult_latex_spec(group):
    assert group in ["A", "B"]
    w = 360
    h = 120

    spec = LATEX_CFG_SPECS["latex_cfg_hard"]
    return dict(
        type="LaTeXDataset",
        latex_cfg=spec["cfg"],
        font=FONT_GROUP_A if group == "A" else FONT_GROUP_B,
        data_config=dict(
            minimal_length=1,
            maximal_length=spec["maximal_length"],
            dpi=200,
            w=w,
            h=h,
        ),
    )


def noisy_difficult_latex_spec(group, *, binarization_threshold, noise_amount):
    spec = difficult_latex_spec(group).copy()
    spec["type"] = "NoisyBinaryLaTeXDataset"
    spec["binarization_threshold"] = binarization_threshold
    spec["noise_amount"] = noise_amount
    spec["technique"] = "flip"
    return spec
