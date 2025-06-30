import torch
import torch.distributed as dist
from memory_efficient_optimizers.utils.dct import dct3_matrix

PROJ_DCT = 'dct'
PROJ_HDM = 'hdm'

ALL_PROJ = [
    PROJ_DCT, # DCT projection
    PROJ_HDM, # Hadamard projection
]

class MatrixStorage:
    """
        This singleton class stores a dictionary where:
        - keys = the matrix size
        - values = the corresponding orthogonal matrix of DCT-3 or Hadamard transforms of size stored in the key
    """
    _instance = None

    @staticmethod
    def init():
        if MatrixStorage._instance is None:
            MatrixStorage._instance = MatrixStorage()

    @staticmethod
    def get_instance():
        if MatrixStorage._instance is None:
            MatrixStorage.init()
        return MatrixStorage._instance

    @staticmethod
    def get_matrix(size, proj, dtype):
        return MatrixStorage.get_instance()._get_matrix(size, proj, dtype)

    @staticmethod
    def add_matrix(size, proj, dtype):
        return MatrixStorage.get_instance()._add_matrix(size, proj, dtype)

    def __init__(self):
        self.storage = dict()
        self.dtype = None
        self.device = f'cuda:{dist.get_rank()}' if dist.is_initialized() else 'cuda:0'

    def _add_matrix(self, size, proj, dtype):
        if size not in self.storage:
            if proj == PROJ_DCT:
                self.storage[size] = dct3_matrix(size).to(device=self.device, dtype=dtype) # first row is zero
            elif proj == PROJ_HDM:
                self.storage[size] = hadamard_transform(torch.eye(size).to(device=self.device, dtype=dtype), scale=1./math.sqrt(size))
            else:
                raise RuntimeError(f'Projection {proj} is currently not supported!')

    def _get_matrix(self, size, proj, dtype):
        if size not in self.storage:
            assert dtype is not None
            self._add_matrix(size, proj, dtype)
        return self.storage[size]
