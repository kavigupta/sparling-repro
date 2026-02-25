import os
import pickle
import struct


class SequentialDiskFile:
    """
    Represents a list of byte blobs as a file on disk, with an associated
    index file that maps indices to byte offsets in the file.
    """

    def __init__(self, path, mode):
        assert mode in {"r", "r+"}
        self.path = path
        self.index_path = path + ".index"

        if not os.path.exists(self.index_path):
            open(self.index_path, "w").close()
            open(self.path, "w").close()

        self.file = open(self.path, mode + "b")
        self.index_file = open(self.index_path, mode + "b")

        self.length = os.path.getsize(self.index_path) // 8

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        assert 0 <= i < len(self)
        self.index_file.seek(i * 8)
        offset = struct.unpack("Q", self.index_file.read(8))[0]
        self.file.seek(offset)
        return pickle.load(self.file)

    def append(self, x):
        self.file.seek(0, 2)
        self.index_file.seek(0, 2)
        offset = self.file.tell()
        pickle.dump(x, self.file)
        self.index_file.write(struct.pack("Q", offset))
        self.length += 1

    def extend(self, xs):
        for x in xs:
            self.append(x)

    def close(self):
        self.file.close()
        self.index_file.close()


class SequentialCacheDiskFile:
    """
    Represents a function f : [0, 1, ..., n] -> picklable object as a file on disk,
        using SequentialDiskFile to store the pickled objects. Expands automatically
        as needed.

    Arguments
    ---------
    path : str
        Path to the file on disk.
    mapped_f: list[int] -> list[object]
        Function representing the underlying function. Should have identical
            semantics to mapped_f = lambda idxs: [f(i) for i in idxs].
    batch_size : int
        Number of indices to evaluate at once. Defaults to 100.
    """

    def __init__(self, path, mapped_f, batch_size=100):
        self.path = path
        self.mapped_f = mapped_f
        self.file = SequentialDiskFile(path, "r+")
        self.batch_size = batch_size

    def __getitem__(self, i):
        assert i >= 0
        while i >= len(self.file):
            inp = range(len(self.file), len(self.file) + self.batch_size)
            out = self.mapped_f(inp)
            self.file.extend(out)
        return self.file[i]
