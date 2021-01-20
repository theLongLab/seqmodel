Code Organization
=================

All code is under `src/seqmodel`. All unit tests are under `test/seqmodel`. Data files included with tests are in `test/data`.

- `data` contains genomic data.
- `src/seqmodel` contains code.
    - `seqmodel.experiment` contains individual experiments and hyperparameters
    - `seqmodel.functional` contains functions for preprocessing data for specific tasks
        - `seqmodel.functional.log` contains functions for summarizing tensors and outputting loggable strings
        - `seqmodel.functional.transform` contains common sequence transformations which are composable
    - `seqmodel.model` contains model components based on `nn.Module`
    - `seqmodel.seqdata` contains functions and objects for loading data
- `test/data` contains data files for tests.
- `test/seqmodel` contains unit tests.


Installation
============

Requires `python 3.6>` and `cuda/10.1`. Uses `pytorch==1.5`. Install dependencies using:
```
pip install --upgrade pip
pip install -r $SOURCE_DIR/requirements.txt
```


Git Guidelines
==============

Developing
----------
- Create and checkout a new branch:
    ```
    git branch feat/[development|experimental]/{feature_name}
    git checkout {branch_name}
    ```
- Develop features in `src/**` and unit tests in `test/**`.
- Commit cohesive units of code in progress:
    ```
    git add -A
    git commit
    ```
    Refer to `git log` for examples of previous commits.
    Follow these guidelines when writing commit messages:
    ```
    {tag [ADD|FIX|DOC|STYLE|REFACTOR|TEST|CHORE]} {summary no more than 50 characters}
    {empty line}
    {tag} {paragraph explaining detail}
    {empty line}
    (optional {tag [TODO|FIXME]} {additional paragraphs for other details. Use TODO to indicate WIP and FIXME to indicate issues.}
    {empty line})
    ```
- Make sure unit tests pass:
    ```
    python -m unittest discover
    ```
    Test integration by running on HPC.
- Once tests pass, merge and fix merge issues:
    ```
    git checkout main
    git merge {branch_name}
    ```

Running on HPC
--------------
- On HPC, create bare repository:
    ```
    mkdir {target_dir} {target_dir}/.git
    cd {target_dir}/.git
    git init --bare
    ```
- On HPC, add post-receive hook to checkout code after pushing.
    ```
    cat >> hooks/post-receivetest << EOL
    #!/bin/sh
    GIT_WORK_TREE={target_dir} git checkout -f

    EOL
    chmod +x hooks/post-receive
    ```
- On local machine, add remote:
    ```
    git remote add {remote_name} {username}@{HPC_url}:{target_dir}/.git
    ```
- To run a particular branch on HPC, checkout and push from the local machine:
    ```
    git checkout {branch_name}
    git push {remote_name} {branch_name}
    ```

Pytorch-Lightning
=================

Default Command Line Arguments
------------------------------
These are for pytorch-lightning only, see individual model files (e.g. `src/exp/seqbert/pretrain.py`) for model-specific arguments.

- accumulate_grad_batches=1
- amp_backend='native'
- amp_level='O2'
- auto_lr_find=False
- auto_scale_batch_size=False
- auto_select_gpus=False
- benchmark=False
- check_val_every_n_epoch=1
- checkpoint_callback=True
- default_root_dir=None
- deterministic=False
- distributed_backend=None
- early_stop_callback=False
- fast_dev_run=False
- gpus=1
- gradient_clip_val=0
- limit_test_batches=1.0
- limit_train_batches=1.0
- limit_val_batches=1.0
- log_gpu_memory=None
- log_save_interval=100
- logger=True
- max_epochs=1000
- max_steps=None
- min_epochs=1
- min_steps=None
- num_nodes=1
- num_processes=1
- num_sanity_val_steps=2
- overfit_batches=0.0
- overfit_pct=None
- precision=32
- prepare_data_per_node=True
- process_position=0
- profiler=None
- progress_bar_refresh_rate=1
- reload_dataloaders_every_epoch=False
- replace_sampler_ddp=True
- resume_from_checkpoint=None
- row_log_interval=50
- sync_batchnorm=False
- terminate_on_nan=False
- test_percent_check=None
- tpu_cores=<function Trainer._gpus_arg_default at 0x7f614e8940d0>
- track_grad_norm=-1
- train_percent_check=None
- truncated_bptt_steps=None
- val_check_interval=1.0
- val_percent_check=None
- weights_save_path=None
- weights_summary='top'
