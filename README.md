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
