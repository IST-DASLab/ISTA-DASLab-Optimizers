class WeightDecayScheduler:
    def __init__(self, weight_decay: float, wd_type: str):
        # assert wd_type in ['const', 'balanced', 'opt-a', 'opt-b']
        self.weight_decay = weight_decay
        self.wd_type = wd_type
        self.awd = 0 # for AWD from apple

    def get_wd(self):
        if self.wd_type == 'const':
            return self.weight_decay
        if self.wd_type == 'awd':
            return self.awd

    def __call__(self, w=None, g=None):
        """
        :param w: tensor that contains weights
        :param g: tensor that contains gradients
        :return: the value for the weight decay
        """
        if self.wd_type == 'const':
            return self.weight_decay

        if self.wd_type == 'awd': # AWD from the Apple paper: https://openreview.net/pdf?id=ajnThDhuq6
            assert (w is not None) and (g is not None),\
                'The balanced weight decay scheduler requires valid for w and g, but at least one is None!'

            # in the paper, lambda_awd is set by the user and they return a lambda_bar
            # here, self.awd will be the moving average from the line 8 in their algorithm and
            # the input to our algorithm is self.weight_decay for all wd_types!
            self.awd = 0.1 * self.awd + 0.9 * self.weight_decay * g.norm(p=2) / w.norm(p=2)
            return self.awd
