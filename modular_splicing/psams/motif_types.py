from .sources.rbns import read_rbns_motifs


def motifs_types():
    return dict(rbns=read_rbns_motifs)
