from dataclasses import dataclass


@dataclass
class PartitionStats:
    R_full: None
    C_full: None
    num_row_blocks: None
    num_col_blocks: None
    num_blocks_full: None
    row_rest: None
    col_rest: None

    def __init__(self,
                 R_full=None,
                 C_full=None,
                 num_row_blocks=None,
                 num_col_blocks=None,
                 num_blocks_full=None,
                 row_rest=None,
                 col_rest=None):
        self.R_full = R_full
        self.C_full = C_full
        self.num_row_blocks = num_row_blocks
        self.num_col_blocks = num_col_blocks
        self.num_blocks_full = num_blocks_full
        self.row_rest = row_rest
        self.col_rest = col_rest
