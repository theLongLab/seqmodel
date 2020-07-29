
class Trainer():

    def __init__(self):
        # logging and checkpoint directory
        # set up SummaryWriter if using tensorboard
        pass

    def train(self, task, data_loader):
        # early stopping
        # parameter updates
        pass

    def evaluate(self, task, valid_loader):
        pass

# epoch loop - epoch
# batch loop - batch, data
# accumulate gradient(task, data) - loss
# report(loss)
# evaluate(task) - accuracy
# snapshot(task) - 
# save checkpoint(task)

class PeriodicAction():

    """
    Wrapper for a function to make it run periodically in a TaskLoop object.
    By default the action occurs on every iteration.
    """
    def __init__(self):
        pass


class TaskLoop():

    """
    Contains list of PeriodicAction.
    Iterates over a loop.
    Runs PeriodicAction and provides a set of arguments to each action.
    """
    def __init__(self):
        pass

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        pass


class FixedLoop(TaskLoop):

    """
    Iterates a fixed number of times.
    """

def loop(self):
    if self._report_iteration_n:
        for i, args in enumerate(self._list_obj):
            yield i, *args
    else:
        for args in self._list_obj:
            yield *args


class EpochLoop(FixedLoop):

    def __init__(self, n_epoch):
        self._list_obj = range(n_epoch)
        self._report_iteration_n = False

    