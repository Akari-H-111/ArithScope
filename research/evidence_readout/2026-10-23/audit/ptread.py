"""Stdlib-only reader for torch .pt zip checkpoints (read-only; tensors -> nested lists of floats)."""
import collections
import pickle
import struct
import zipfile

_DT = {"FloatStorage": ("f", 4), "DoubleStorage": ("d", 8), "LongStorage": ("q", 8),
       "IntStorage": ("i", 4), "BoolStorage": ("?", 1)}


class _Storage:
    def __init__(self, kind, key):
        self.kind, self.key = kind, key


class _Tensor:
    def __init__(self, st, off, size, stride):
        self.st, self.off, self.size, self.stride = st, off, tuple(size), tuple(stride)


def _rebuild(storage, off, size, stride, *a, **k):
    return _Tensor(storage, off, size, stride)


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == "_rebuild_tensor_v2":
            return _rebuild
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if mod == "torch" and name in _DT:
            return name
        if mod.startswith("torch") or mod.startswith("numpy"):
            return lambda *a, **k: (mod, name, a)
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        return _Storage(pid[1], pid[2])


def load(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    pkl = [n for n in names if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]
    obj = _U(z.open(pkl)).load()

    def to_list(t):
        code, sz = _DT[t.st.kind]
        raw = z.read(prefix + "data/" + str(t.st.key))
        n = len(raw) // sz
        flat = struct.unpack("<%d%s" % (n, code), raw)

        def rec(dim, base):
            if dim == len(t.size):
                return flat[base]
            return [rec(dim + 1, base + i * t.stride[dim]) for i in range(t.size[dim])]
        return rec(0, t.off)

    def conv(o):
        if isinstance(o, _Tensor):
            return to_list(o)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    return conv(obj)
