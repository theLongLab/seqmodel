# Link to this document: https://drive.google.com/file/d/1ZvsKTM2uYNkDnjzFWbIJl1awM58lE2gz/view?usp=sharing

# Alternatives
#     kipoi
#         Example: https://github.com/kipoi/models/tree/master/DeepSEA
#         Dataloader: https://github.com/kipoi/kipoiseq/blob/master/kipoiseq/dataloaders/sequence.py
#         can't load arbitrary sequences from intervals, need to explicitly indicate intervals
#     selene - limited to pytorch 1.4, doesn't retrieve the data automatically
#         Example: https://github.com/FunctionLab/selene/blob/master/tutorials/getting_started_with_selene/getting_started_with_selene.ipynb
# advantages of torchvision
#     Example: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.htmlsphx-glr-beginner-blitz-cifar10-tutorial-py
#     mostly functional approach for transforming data
#     one api for both manipulating and downloading data - easier to learn and pipeline than multiple command line tools

# whether to use pytorch-lightning?
#     convenient, less engineering for boring stuff, less bugs?
#     easy to transition to own training loop given same model organization

# functional approach, vs all-in-one objects
#     functions are easy to compose and modular
#     all-in-one objects are good for models which are highly variable, keeping them self-contained and easy to track changes
#     avoid multiple interacting objects where not needed, as makes code rigid and hard to apply elsewhere
#     be careful about dimensions of data, sequence, batch, and channel dimensions aren't the same in all pytorch API

# pyfaidx vs BioPython
#     pyfaidx allows streaming fasta files, but only one person maintaining?
#     BioPython is more comprehensive, but is it being maintained?

# unit testing
#     dev projects are slow
#     modular is good, because git merges based on whole text files and sections of text

# walk through entire BERT model
# https://github.com/theLongLab/seqmodel/blob/feat/development/datasets/src/experiment/seqbert.py

# A quick Git primer.
#     everyone has their own copy
#     keep the main branch clean, don't merge WIP

# Sample development procedure:
# ```


# Github hosts many open source projects including pytorch, tensorflow, keras...
    # Navigate to https://github.com/theLongLab/seqmodel
    # Note the different branches available in the top left, these are like versions
    # The URL for cloning is on the right

# Let's clone the repo: browse git@github.com:theLongLab/seqmodel.git
mkdir seqmodel-example
cd !$
git clone -b feat/development/datasets https://github.com/theLongLab/seqmodel.git .
    # clone branch to current directory '.'
git log
    # pageup/pagedn to navigate, press q to quit
    # note the commit hashes
git log --graph --oneline --all
    # pretty print summary

# We need to install all prerequisites to develop
    # A virtual environment is a 'clean slate' in we can add/remove
    # dependencies without affecting the rest of the system
mkdir seqmodel-env
    # save dependencies here
virtualenv --python=/usr/bin/python3 seqmodel-env
    # replace /usr/bin/python3 with the install location of the desired python version
source seqmodel-env/bin/activate
    # activate the environment
python --version
    # normally, if python 2 and python 3 are both installed
    # this will open the python 2 interpreter
    # but the virtual environment only has python 3 installed

# Install everything listed in requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

# Quickest way to check everything works is using the unit tests
python -m unittest discover
    # note: this only finds unittests that are in the import path (looks for __init__.py)
    # run in the root dir of the project

# Let's make a new branch to develop on
git branch
    # lists branches
git branch feat/experimental/2020-09-01-example
    # creates the branch
    # branches can be named just like directories/files
    # however, leaf nodes cannot be extended:
    # e.g. main is a branch, main/something cannot be
git checkout !$

# Try modifying the hyperparameters in `train.seqbert.py`
# If you forgot to checkout the new branch before making changes
    # temporarily store the changes using `git stash`
    # checkout the new branch, and apply the changes with `git stash pop`

# Once finished, commit changes to the current branch
python -m unittest discover
    # Before committing, always check that no unit tests broke
git add -A
# Follow the guidelines in `README.md` when writing commit messages
    # use -m to write short message
    # git will open default text editor otherwise
git commit -m "MODIFY seqbert hparams, remove redundant"

# To push changes to a new destination, we need to add a new remote location
git remote -v
    # lists remotes, -v shows the URLs
git remote hpc user@hpc:some/dir
    # example: add remote called 'hpc' to computing cluster

# need to create a repository at the remote location
ssh user@hpc
mkdir some/dir
cd !$
git init --bare .git
    # initialize bare repo in .git (replace .git with any directory)
    # a bare repo means we don't do any development here
    # so no files have to be checked out
# add hook, which is run after receiving a push
nano hooks/post-receive  # use text editor of your choice, nano is commonly installed
#!/bin/sh
GIT_WORK_TREE=~/some/dir git checkout -f
    # copy the above into the text file
    # this command will automatically checkout whatever was pushed to the target dir
    # Ctrl+X, Y to save and exit
chmod +x hooks/post-receive
    # make hook executable
echo ref: refs/heads/main > HEAD
    # need to set default branch to `main`, because normally the default is `master`
exit
    # leave remote machine

# now we can push from the local machine
git push hpc feat/experimental/2020-09-01-example
    # push needs to name the specific branch that you want deployed
    # otherwise it will deploy main
ssh user@hpc
cd some/dir
ls
    # check that files went through
cat train-seqbert.sh
    # check that updates went through
    # can also use `rsync`, or manually checkout using the command in `hooks/post-receive`

# Merging a finished branch
git checkout main
    # checkout target branch
git merge exp/seqbert/2020-09-01-example
    # merge the source branch
    # If there are merge conflicts, go through each conflict
    # pick which version to keep, then add and commit the resulting files

# Finally to clean up
deactivate
    # turn off the virtual env
git reset HEAD~
    # this can undo last commit
    #HEAD is a special pointer to the current position of the tree
    #HEAD~ goes back one node
rm seqmodel-example -dr
    # removing the entire directory will cleanly remove all code,
    # git objects, and dependencies
    # this is the nuclear option for fixing a bad repo

#```
