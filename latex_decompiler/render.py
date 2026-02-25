import os
import subprocess
import tempfile
from colorsys import hsv_to_rgb
from textwrap import dedent

import numpy as np
import pdf2image
from permacache import permacache

TEMPLATE = r"""
\documentclass[border=3pt]{standalone}
\usepackage{amsmath}
\usepackage{xcolor}

% to let you color just the left or right delimiter
% from https://tex.stackexchange.com/a/195274/43770
\newcommand{\cleft}[2][.]{%
  \begingroup\colorlet{savedleftcolor}{.}%
  \color{#1}\left#2\color{savedleftcolor}%
}
\newcommand{\cright}[2][.]{%
  \color{#1}\right#2\endgroup
}

COLORS
FONTS
\begin{document}
$\displaystyle
EQN
$
\end{document}
"""


def antykwa_torunska(option):
    return r"\usepackage[math$X]{anttor}".replace("$X", option)


def baskervaldx():
    return dedent(
        r"""
        \usepackage[lf]{Baskervaldx} % lining figures
        \usepackage[bigdelims,vvarbb]{newtxmath} % math italic letters from Nimbus Roman
        \usepackage[cal=boondoxo]{mathalfa} % mathcal from STIX, unslanted a bit
        \renewcommand*\oldstylenums[1]{\textosf{#1}}
        """
    )


def baskervillef():
    return dedent(
        r"""
        \usepackage[T1]{fontenc}
        \usepackage{baskervillef}
        \usepackage[varqu,varl,var0]{inconsolata}
        \usepackage[scale=.95,type1]{cabin}
        \usepackage[baskerville,vvarbb]{newtxmath}
        \usepackage[cal=boondoxo]{mathalfa}
        """
    )


def fira_sans_newtxsf():
    return dedent(
        r"""
        \usepackage[sfdefault,scaled=.85]{FiraSans}
        \usepackage{newtxsf}
        """
    )


def heuristica():
    return dedent(
        r"""
        \usepackage{heuristica}
        \usepackage[heuristica,vvarbb,bigdelims]{newtxmath}
        \usepackage[T1]{fontenc}
        \renewcommand*\oldstylenums[1]{\textosf{#1}}
        """
    )


def kp_sans():
    return dedent(
        r"""
        \usepackage[sfmath]{kpfonts} %% sfmath option only to make math in sans serif. Probablye only for use when base font is sans serif.
        \renewcommand*\familydefault{\sfdefault} %% Only if the base font of the document is to be sans serif
        """
    )


FONT_TO_HEADER = {
    "times": r"\usepackage{newtxmath}",
    "antykwa_torunska": antykwa_torunska(""),
    "antykwa_torunska_condensed": antykwa_torunska(",condensed"),
    "antykwa_torunska_light": antykwa_torunska(",light"),
    "antykwa_torunska_light_condensed": antykwa_torunska(",light,condensed"),
    "arev": r"\usepackage{arev}",
    "gfs_artemisia": r"\usepackage{gfsartemisia}",
    "gfs_artemisia_euler": r"\usepackage{gfsartemisia-euler}",
    # "asana_math": r"\usepackage{fontspec}" + "\n" + r"\setmainfont{Asana-Math}",
    "baskervaldx": baskervaldx(),
    "baskervillef": baskervillef(),
    "boisik": r"\usepackage{boisik}",
    "charterbt": r"\usepackage[bitstream-charter]{mathdesign}",
    "concmath": r"\usepackage{concmath}",
    "concmath_euler": r"\usepackage{beton,euler}",
    "computer_modern_bright": r"\usepackage{cmbright}",
    "computer_modern": "",
    "euler": r"\usepackage{euler}",
    "fira_sans_newtxsf": fira_sans_newtxsf(),
    # "garamond": r"\usepackage[urw-garamond]{mathdesign}",
    "eb_garabond": r"\usepackage[cmintegrals,cmbraces]{newtxmath}"
    + "\n"
    + r"\usepackage{ebgaramond-maths}",
    # "garamond_expert": r"\usepackage[urw-garamond]{mathdesign}" + "\n" + r"\usepackage{garamondx}",
    "heuristica": heuristica(),
    "iwona": r"\usepackage[math]{iwona}",
    "iwona_condensed": r"\usepackage[condensed,math]{iwona}",
    "iwona_light": r"\usepackage[light,math]{iwona}",
    "iwona_light_condensed": r"\usepackage[light,condensed,math]{iwona}",
    "kerkis": r"\usepackage{kmath,kerkis}",
    "kp_sans": kp_sans(),
    "kp_serif": r"\usepackage{kpfonts}",
    "kurier": r"\usepackage[math]{kurier}",
    "kurier_light": r"\usepackage[light,math]{kurier}",
    "kurier_light_condensed": r"\usepackage[light,condensed,math]{kurier}",
    "lmodern": r"\usepackage{lmodern}",
    "libertinus": r"\usepackage{libertinus}",
    "lxfonts": r"\usepackage{lxfonts}",
    "mlmodern": r"\usepackage{mlmodern}",
    "gfs_neohellenic": r"\usepackage[default]{gfsneohellenic}",
    # "gfs_neohellenic_math": r"\usepackage{gfsneohellenicot}",
    "new_px": r"\usepackage{newpxtext,newpxmath}",
    "new_px_euler": r"\usepackage{newpxtext,eulerpx}",
    "newtx": r"\usepackage{newtxtext,newtxmath}",
    "urw_nimbus_roman": r"\usepackage{mathptmx}",
    "noto_serif": r"\usepackage{notomath}",
    "urw_palladio": r"\usepackage{mathpazo}" + "\n" + r"\linespread{1.05}",
    "pxfonts": r"\usepackage{pxfonts}",
    # scholax
    "urw_schoolbook_l": r"\usepackage{fouriernc}",
    "step": r"\usepackage[notext]{stix}" + "\n" + r"\usepackage{step}",
    "stickstoo": r"\usepackage{stickstootext}"
    + "\n"
    + r"\usepackage[stickstoo,vvarbb]{newtxmath}",
    "stix": r"\usepackage{stix}",
    "stix2": r"\usepackage{stix2}",
    "txfonts": r"\usepackage{txfonts}",
    "utopia_fourier": r"\usepackage{fourier}",
    "utopia_mathdesign": r"\usepackage[adobe-utopia]{mathdesign}",
    # "xits": dedent(
    #     r"""
    #     \usepackage{unicode-math}
    #     \setmainfont{XITS}
    #     \setmathfont{XITS Math}
    #     """
    # ),
}

SINGLE_CHARACTER_REPRESENTATIONS = {
    "FRACBABR": [r"\frac{" + "\mbox{ }" * i + "}{\mbox{ }}" for i in range(1, 1 + 4)]
}

hues = range(0, 360, 20)
hue_names = ["color" + str(hue) for hue in hues]


def color_definitions():
    return "\n".join(
        r"\definecolor{$name}{rgb}{$rgb}".replace("$name", name).replace(
            "$rgb", ",".join(map(str, hsv_to_rgb(hue / 360, 1, 1)))
        )
        for name, hue in zip(hue_names, hues)
    )


def render(equation, font, dpi=200):
    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "eqn.tex"), "w") as f:
            f.write(
                TEMPLATE.replace("EQN", equation)
                .replace("COLORS", color_definitions())
                .replace("FONTS", FONT_TO_HEADER[font])
            )
        subprocess.check_call(
            ["pdflatex", "eqn.tex"],
            cwd=directory,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        [image] = pdf2image.convert_from_path(
            os.path.join(directory, "eqn.pdf"), dpi=dpi
        )
    return image


@permacache("latex_decompiler/render/clipped_image")
def clipped_image(text, font, dpi):
    image = np.array(render(text, font, dpi=dpi))
    ys, xs = np.where((image != 255).any(-1))
    image = image[ys.min() : 1 + ys.max(), xs.min() : 1 + xs.max(), 0]
    return image


def character_representations(character):
    if character in SINGLE_CHARACTER_REPRESENTATIONS:
        return SINGLE_CHARACTER_REPRESENTATIONS[character]
    assert len(character) == 1
    return [character]
