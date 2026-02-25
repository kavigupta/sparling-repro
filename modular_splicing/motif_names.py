from dconstruct import construct

from modular_splicing.psams.motif_types import motifs_types


def get_motif_names(motif_names_source):
    assert motif_names_source == "rbns"
    return sorted(construct(motifs_types(), dict(type="rbns")))[2:]
