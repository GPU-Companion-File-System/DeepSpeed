import os

from .builder import CUDAOpBuilder

def detect_sm_arch():
    import torch
    """Return best compute capability, e.g. 89 for L40S."""
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor

class GeminiFSBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_GEMINIFS"
    NAME = "geminifs"

    def __init__(self, name=None):
        super().__init__(name=self.NAME if name is None else name)
        self.geminifs_path =  os.path.dirname(os.path.abspath(__file__))
        retry = 10
        while retry > 0:
            if os.path.basename(self.geminifs_path) != "DeepSpeed":
                self.geminifs_path = os.path.dirname(self.geminifs_path)
                retry -= 1
            else:
                self.geminifs_path = os.path.join(self.geminifs_path, "../Geminifs")
                break
        
        self.sm_arch = detect_sm_arch()
        if self.sm_arch is None:
            self.warning("No CUDA device detected, cannot determine sm version")
        else:
            self.warning(f"Detected GPU sm_{self.sm_arch}")

    def absolute_name(self):
        return f'deepspeed.ops.geminifs.{self.NAME}_op'

    def sources(self):
        return [ 'csrc/geminifs/py_lib/ds_py_geminifs_handle.cu', 
                 'csrc/geminifs/py_lib/py_ds_geminifs.cu']

    def cxx_args(self):
        geminifs_lib_path = os.path.join(self.geminifs_path, "build", "lib")
        return super().cxx_args() + [f"-L{geminifs_lib_path}"]

    def nvcc_args(self):
        base = super().nvcc_args() + ['-O3'] + self.version_dependent_macros()
        base = [arg for arg in base if arg.startswith("-gencode=") is False]

        sm = self.sm_arch if self.sm_arch is not None else 90
        geminifs_lib_path = os.path.join(self.geminifs_path, "build", "lib")
        nvcc_extra = [
            f"-L{geminifs_lib_path}",
            f"-gencode=arch=compute_{sm},code=sm_{sm}",
            "-lgeminifs",
        ]
        print("NVCC ARGS:", base + nvcc_extra)

        return base + nvcc_extra

    def include_paths(self):
        GEMINIFS_INCLUDE = os.path.join(self.geminifs_path, "libgeminifs", "include")
        LIBNVME_INCLUDE = os.path.join(self.geminifs_path, "libnvm", "include")
        return [GEMINIFS_INCLUDE, LIBNVME_INCLUDE]

    def extra_ldflags(self):
        geminifs_lib_path = os.path.join(self.geminifs_path, "build", "lib")
        return super().extra_ldflags() + ['-lgeminifs', f'-L{geminifs_lib_path}']

    def is_compatible(self, verbose=False):
        try:
            import torch.utils.cpp_extension
        except ImportError:
            if verbose:
                self.warning("Please install torch if trying to pre-compile GeminiFS")
            return False

        GEMINIFS_LIB = os.path.join(self.geminifs_path, "build", "lib")

        if not os.path.exists(GEMINIFS_LIB):
            self.warning(f"GEMINIFS_LIB not found in expected path: {GEMINIFS_LIB}")
            GEMINIFS_LIB = None
            return False

        return super().is_compatible(verbose)
