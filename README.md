loader
    - manages hyperparameters and default states for optuna
    - takes configs
    - creates objects
trainer
    - splits data, sends to task
    - data goes to GPU here
    - reports metrics
    - saves checkpoints
task
    - dictates data required
    - preprocesses data
    - runs through model
    - gets loss
    loss
        - plug and play modules
model
    - same as current
seq
    - plug and play datasets, loaders