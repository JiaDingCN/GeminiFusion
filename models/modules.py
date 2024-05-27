import torch.nn as nn

num_parallel = 2





class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class Additional_One_ModuleParallel(nn.Module):
    def __init__(self, module):
        super(Additional_One_ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel, x_arg):
        if x_arg == None:
            return [self.module(x, None) for x in x_parallel]
        elif isinstance(x_arg, list):
            return [
                self.module(x_parallel[i], x_arg[i]) for i in range(len(x_parallel))
            ]
        else:
            return [self.module(x_parallel[i], x_arg) for i in range(len(x_parallel))]


class Additional_Two_ModuleParallel(nn.Module):
    def __init__(self, module):
        super(Additional_Two_ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel, x_arg1, x_arg2):
        return [
            self.module(x_parallel[i], x_arg1, x_arg2) for i in range(len(x_parallel))
        ]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, "ln_" + str(i), nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, "ln_" + str(i))(x) for i, x in enumerate(x_parallel)]
