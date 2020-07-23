

class ModuleUnit(nn.Module):

    def __init__(self, activation_fn):
        pass

    # number of channels per sequence position
    @property
    def out_channels(self):
        pass

    def forward(self, x):
        pass


class ConvUnit(ModuleUnit):

    def __init__(self):
        pass

    def forward(self, x):
        pass


class ConvEncoder(ModuleUnit):

    def __init__(self):
        pass

    def forward(self, inputs, targets):
        pass