"""Keep optional binary parser imports free of subprocess probes."""


def prepare():
    import platform

    # h5py's binary extensions index platform.uname(), which in Python 3.12
    # evaluates the lazy `processor` field and otherwise runs `uname -p`.
    # The architecture from os.uname is already in `machine`; fill the cache
    # without invoking that property. This is informational parser metadata.
    info = platform.uname()
    if "processor" not in vars(info):
        info.processor = info.machine
