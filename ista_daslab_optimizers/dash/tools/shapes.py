from dataclasses import dataclass

from ..types import DashPartitionInfo
from .stats import PartitionStats


@dataclass
class DashShape3D:
    # __slots__ = ("b", "m", "n")
    """
    This is a 3-tuple that overloads the + and += operators to be able to do (A, B, C) + (D, B, C) = (A+D, B, C)
    It is useful in adding shapes when we merge parameters and allows operations like (A, B, C) + None = (A, B, C)
    """
    b: int
    m: int
    n: int

    # ADDITION
    def __add__(self, other): # overload the + operator
        if other is None: return self
        assert self.m == other.m and self.n == other.n
        if not isinstance(other, DashShape3D): return NotImplemented
        return DashShape3D(self.b + other.b, self.m, self.n)

    def __radd__(self, other):
        if other is None:
            return self
        return NotImplemented

    def __iadd__(self, other): # overload the += operator
        return self + other

    # MULTIPLICATION
    def __mul__(self, k):
        if isinstance(k, int):
            return DashShape3D(self.b * k, self.m, self.n)
        return NotImplemented

    def __rmul__(self, k):
        # allows: k * DashShape3D(...)
        return self.__mul__(k)

    def __imul__(self, k):
        # immutable semantics
        return self * k

    # DIVISON
    def __truediv__(self, k):
        """True division: returns float in first dimension"""
        if isinstance(k, (int, float)):
            return DashShape3D(self.b / k, self.m, self.n)
        return NotImplemented

    def __floordiv__(self, k):
        """Floor division: returns integer in first dimension"""
        if isinstance(k, int):
            return DashShape3D(self.b // k, self.m, self.n)
        return NotImplemented

    def __itruediv__(self, other):
        return self / other

    def __ifloordiv__(self, other):
        return self // other

    def __rtruediv__(self, k):
        return NotImplemented  # usually not defined

    def __rfloordiv__(self, k):
        return NotImplemented  # usually not defined

    # INDEXING
    def __getitem__(self, idx):
        return (self.b, self.m, self.n)[idx]
        # if idx == 0:
        #     return self.b
        # elif idx == 1:
        #     return self.m
        # elif idx == 2:
        #     return self.n
        # else:
        #     raise IndexError("DashShape3D index out of range")

    def as_tuple(self):
        return (self.b, self.m, self.n)

    @classmethod
    def like(cls, other):
        return cls(other.b, other.m, other.n)


@dataclass
class DashMultiShape:
    # example for embedding size (32_000, 2048) and block size B=1024
    Gfull:  None # ( 62, 1024, 1024)
    Grest:  None # (  2,  256, 1024)
    Lfull:  None # ( 62, 1024, 1024): Gfull        @ Gfull.T(1,2)
    Rfull:  None # ( 62, 1024, 1024): Gfull.T(1,2) @ Gfull
    Lrest:  None # (  2,  256,  256): Grest        @ Grest.T(1,2)
    Rrest:  None # (  2, 1024, 1024): Grest.T(1,2) @ Grest
    LRfull: None # (126, 1024, 1024): stacks Lfull, Rfull, Rrest
    LRrest: None # (  2,  256,  256): this is Lrest
    stats:  None # stats used to populate the gradient block (Rfull, Cfull, num blocks and others)
    info:   None # DashDashPartitionInfo.EFFICIENT_BLOCK_GROUPING_FULL_LR_REST_R_AND_REST_L

    def __init__(self, Gfull, Grest, stats, is_norm_layer_stack): # Lfull, Rfull, Lrest, Rrest, DASHfull, DASHrest, info):
        self.Gfull = Gfull
        self.Grest = Grest
        # self.Lfull = Lfull
        # self.Rfull = Rfull
        # self.Lrest = Lrest
        # self.Rrest = Rrest
        # self.LRfull = LRfull
        # self.LRrest = LRrest
        # self.info = info
        self.stats = stats
        self.is_norm_layer_stack = is_norm_layer_stack
        self.has_rest = Grest is not None
        self._compute_L_R_shapes()

    def _compute_L_R_shapes(self):
        Gfull = self.Gfull
        Grest = self.Grest
        self.Lfull = DashShape3D(Gfull[0], Gfull[1], Gfull[1])
        self.Rfull = DashShape3D(Gfull[0], Gfull[2], Gfull[2])
        if self.is_norm_layer_stack: # Normalization Layer
            self.Lrest = None
            self.Rrest = None
            self.LRfull = DashShape3D.like(self.Lfull) # (Gfull[0], Gfull[1], Gfull[1])
            self.LRrest = None
            self.info = DashPartitionInfo.REGULAR_BLOCK
        else: # Linear Layer
            if Grest is None:
                self.Lrest = None
                self.Rrest = None
                self.info = DashPartitionInfo.NO_REST
                self.LRfull = self.Lfull + self.Rfull # DashShape3D(2 * Gfull[0], Gfull[1], Gfull[2])
                self.LRrest = None
            else:
                self.LRfull = 2 * Gfull # DashShape3D(2 * Gfull[0] + Grest[0], Gfull[1], Gfull[2])
                self.Lrest = DashShape3D(Grest[0], Grest[1], Grest[1])
                self.Rrest = DashShape3D(Grest[0], Grest[2], Grest[2])
                if Gfull[2] == Grest[2]: # it means Lrest has shape (*, b, b), with b < B ===> DASHrest is Lrest
                    self.info = DashPartitionInfo.REST_L
                    self.LRfull += self.Rrest
                    self.LRrest = DashShape3D(Grest[0], Grest[1], Grest[1])
                elif Gfull[1] == Grest[1]: # it means Rrest has shape (*, b, b) with b < B ===> DASHrest is Rrest
                    self.info = DashPartitionInfo.REST_R
                    self.LRfull += self.Lrest
                    self.LRrest = DashShape3D(Grest[0], Grest[2], Grest[2])

    # def __add__(self, other):
    #     # Gfull: None  # ( 62, 1024, 1024)
    #     # Grest: None  # (  2,  256, 1024)
    #     # Lfull: None  # ( 62, 1024, 1024): Gfull        @ Gfull.T(1,2)
    #     # Rfull: None  # ( 62, 1024, 1024): Gfull.T(1,2) @ Gfull
    #     # Lrest: None  # (  2,  256,  256): Grest        @ Grest.T(1,2)
    #     # Rrest: None  # (  2, 1024, 1024): Grest.T(1,2) @ Grest
    #     # LRfull: None  # (126, 1024, 1024): stacks Lfull, Rfull, Rrest
    #     # LRrest: None  # (  2,  256,  256): this is Lrest
    #     # stats: None  # stats used to populate the gradient block (Rfull, Cfull, num blocks and others)
    #     # info: None  # DashDashPartitionInfo.EFFICIENT_BLOCK_GROUPING_FULL_LR_REST_R_AND_REST_L
    #
    #     return DashMultiShape(
    #         Gfull=self.Gfull + other.Gfull,
    #         Gfull=self.Gfull + other.Gfull,
    #         Grest=self.Grest + other.Grest,
    #         Lfull=self.Lfull + other.Lfull,
    #         Rfull=self.Rfull + other.Rfull,
    #         Lrest=self.Lrest + other.Lrest,
    #         Rrest=self.Rrest + other.Rrest,
    #         LRfull=self.LRfull + other.LRfull,
    #         LRrest=self.LRrest + other.LRrest,
    #         stats=self.stats + other.stats,
    #         info=self.info + other.info,
    #     )


class DashShapesCalculator:
    ########################################
    ########## FUNCTIONS FOR NORM LAYERS
    ########################################
    @staticmethod
    def get_stacked_shapes_for_merged_norm_layers(shape, B):
        """
        Given a stacked parameter of shape `shape` (e.g. multiple normalization layers of shape (E,) each) stacked into shape (N, E),
        compute the shape of the blockified parameter:

        For B=1024 and (N=33, E=2048), the gradient shape will be (66, 1024, 1)

        Arguments:
            shaoe (2-tuple): stacked 1D parameters
            B (int): block size
        Returns:
            DashShape object: given the gradient shape (66, 1024, 1), the DashShape object automatically computes outer product shapes
        """
        N = shape[0]
        E = shape[1]
        assert E % B == 0

        R_full = (E // B) * B # store for populating
        num_row_blocks = R_full // B
        num_blocks_full = N * num_row_blocks

        row_rest = E - R_full

        Gfull = DashShape3D(num_blocks_full, B, 1)

        if row_rest > 0: # this is not allowed because of assert at the top
            Grest = DashShape3D(N * 1, row_rest, 1)
        else:
            Grest = None

        return DashMultiShape(Gfull=Gfull,
                              Grest=Grest,
                              stats=PartitionStats(R_full=R_full,
                                                   num_row_blocks=num_row_blocks,
                                                   num_blocks_full=num_blocks_full,
                                                   row_rest=row_rest, ),
                              is_norm_layer_stack=True)

    @staticmethod
    def get_param_shape_of_merged_norm_layers(bucket_func):
        """
        Given a generator `bucket_func` that contains `N` normalization layers of shape (E,),
        this function returns the blockified shape for the gradient.
        For example, if (N,E) = (33, 2048), this function returns (N, E, 1).
        """
        pool = [p for index, group, state, p in bucket_func()]
        N = len(pool)  # number of normalization layers
        E = pool[0].shape[0]  # embedding size
        return DashShape3D(N, E, 1)

    ########################################
    ########## FUNCTIONS FOR LINEAR LAYERS
    ########################################
    @staticmethod
    def get_stacked_shapes_per_single_linear_layer(shape, B):
        """
        Given a 2D layer with shape `shape`, compute the Gfull, Grest and all other shapes automatically
        in the DashShape object.
        """
        R, C = shape
        assert (R % B == 0) or (C % B == 0), f"Block size must be a multiple of rows and columns: {R=}, {C=}, {B=}"

        # Compute full block extents
        R_full = (R // B) * B # store for populating
        C_full = (C // B) * B # store for populating

        num_row_blocks = R_full // B # store for populating
        num_col_blocks = C_full // B # store for populating
        num_blocks_full = num_row_blocks * num_col_blocks # store for populating

        # Remainders
        row_rest = R - R_full # store for populating
        col_rest = C - C_full # store for populating

        shape_full = DashShape3D(num_blocks_full, B, B)# changed!
        if col_rest > 0:
            shape_rest = DashShape3D(num_row_blocks, B, col_rest)
        elif row_rest > 0:
            shape_rest = DashShape3D(num_col_blocks, row_rest, B)
        else:
            shape_rest = None

        return DashMultiShape(Gfull=shape_full,
                              Grest=shape_rest,
                              stats=PartitionStats(R_full=R_full,
                                                   C_full=C_full,
                                                   num_row_blocks=num_row_blocks,
                                                   num_col_blocks=num_col_blocks,
                                                   num_blocks_full=num_blocks_full,
                                                   row_rest=row_rest,
                                                   col_rest=col_rest, ),
                              is_norm_layer_stack=False)

    def get_stacked_shape_for_all_linear_layers(bucket_func, B):
        """
        This function sums up the batch dimensions of all DashShape objects in the state field of params p.
        For the full blocks, we can maintain only one variable to count batch sizes
        For the rest blocks, different layers on a GPU can have different rests b. Therefore, we need a
        dictionary with keys=b and value=sum_of_batch_sizes
        """
        # Gfull = DashShape3D(0, B, B)
        # Grest = {}
        G = {
            (B, B): DashShape3D(0, B, B)
        }

        LR = { # dictionary where key=b or B and value = N: b is the rest size (b < B) and N is the 0-th (batch) dimension of DASHrest
            (B, B): DashShape3D(0, B, B), # will add shapes to the batch dimension
        }
        for index, group, state, p in bucket_func():
            pshape = DashShapesCalculator.get_stacked_shapes_per_single_linear_layer(shape=p.shape, B=B)
            # Gfull += pshape.Gfull # adds only batch dimensions
            # LRfull += pshape.LRfull # adds only batch dimensions
            G[(B, B)] += pshape.Gfull
            LR[(B, B)] += pshape.LRfull

            if pshape.has_rest:
                keyGrest = pshape.Grest[1:] # last two dimensions of the shape
                keyLR = pshape.LRrest[1:] # has shape (n, b, b) and choose last 'b'

                if keyGrest not in G:
                    G[keyGrest] = pshape.Grest
                else:
                    G[keyGrest] += pshape.Grest

                if keyLR not in LR:
                    LR[keyLR] = pshape.LRrest
                else:
                    LR[keyLR] += pshape.LRrest
        # end for-loop

        # return Gfull, Grest, LRfull, LRrest
        # return Gfull, Grest, LR
        return G, LR
