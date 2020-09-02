Installation
============

Requires `python 3.6>` and `cuda/10.1`. Uses `pytorch==1.5` (need to check if `pytorch 1.6>` will work with masking arguments for `torch.nn.TransformerEncoder`). Install dependencies using:
```
pip install --upgrade pip
pip install -r $SOURCE_DIR/requirements.txt
```


Code Organization
=================

All code is under `src/seqmodel`. All unit tests are under `test/seqmodel`. Data files included with tests are in `test/data`.

- `data`: directory containing custom data.
- `.cache`: default directory for automatically downloaded data.
- `src/experiment`: individual experiments and their hyperparameters
    - models follow the `pytorch-lightning` format
    - models do not need unit tests
    - keep model architecture explicit in code, and record hyperparameters in shell scripts (e.g. `train.sh` to allow versioning with git
- `src/seqmodel`: framework code.
    - `seqmodel.functional`: functions for preprocessing data for specific training tasks.
        - `seqmodel.functional.log`: functions for summarizing tensors and outputting loggable strings.
        - `seqmodel.functional.transform`: common sequence transformations which are composable.
    - `seqmodel.model` model components based on `torch.nn.Module`.
    - `seqmodel.seqdata` functions and objects for iterating over multiple types of data. API should mimic `torchvision`.
        - `seqmodel.seqdata.datasets` download and read consortium data of different types.
- `test/data`: data files for tests.
- `test/seqmodel`: unit tests, following same directory structure as `src/seqmodel`.


Git Guidelines
==============

Developing
----------
- Create and checkout a new branch:
    ```
    git branch [model|feat]/[development|experimental]/{feature_name}
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
- Make sure unit tests pass (make sure `__init__.py` exists in every subdirectory containing testing code):
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

Data Pipeline
=============

Data can come from multiple sources (species or individuals) and multiple types (sequences, intervals, variants, phenotypic information). `seqmodel.seqdata.datasets` loads individual types of data, whereas `seqmodel.seqdata` combines multiple types for training.


Datasets
========

Reference Genomes (Genome Research Consortium)
----------------------------------------------

Genome assemblies (containing multiple sequences for a given species) are indexed according to GenBank assembly accession (e.g. human reference genome GRCh38.p13 is `GCA_000001405.28`). Individual sequences are described by their GenBank accession and version (e.g. version 2 of chromosome 1 of human genome is `CM000663.2`)

Sequence data is in compressed text format (gzipped Fasta files `.fna.gz`). This can be read using `pyfaidx.Fasta` after extracting. Annotations and regions are in tab delimited tables (text files) with header indicated by `#`, and can be read using `pandas.read_csv`.

Files are of the format `{accession}_{sequence_name}_{file_type}`. For reference genomes, many of the files containing annotated sequences are very brief, and the majority of the data is in the un-annotated sequence assemblies. List of files types and their uses (**bold** files are relevant):

- **`*_assembly_report.txt` tab delimited table of region names, start/end positions, accession numbers. Use this to map common names to accession (e.g. chromosome `1` to `CM000663.2`). File includes multi-line header indicated by `#`. Columns are:**

    0. **Sequence-Name**: common name (e.g. `Y` for chromosome Y) expose this name to API for end-user's convenience.
    1. **Sequence-Role**: use this to filter sequences (e.g. omit sequences without coordinates).
    2. Assigned-Molecule: Sequence-Name of parent sequence.
    3. Assigned-Molecule-Location/Type: type of parent sequence (e.g. `Chromosome`).
    4. **GenBank-Accn**: GenBank accession identifier (e.g. ``CM000663.2`). Expose to API.
    5. Relationship: not sure what this means. May indicate whether GenBank and RefSeq versions are identical.
    6. RefSeq-Accn: RefSeq accession identifier, an alternate identifier (e.g. chromosome 1 is `NC_000001.11`). Expose to API.
    7. Assembly-Unit: indicates whether sequence is part of the primary assembly, an alternate assembly, or a patch. Can also be used to filter sequences.
    8. **Sequence-Length**: length of sequence in base pairs. Use this to generate `fai` files for `pyfaidx` without iterating over fasta files.
    9. UCSC-style-name: another alternate name (e.g. `chr1`), expose this name to API also.

- `*_assembly_regions.txt` similar structure as `*_assembly_report.txt` but lists alternate sequences. Use this to map names to accession for alternate sequences. Not always available.
- `*_assembly_stats.txt` statistics for evaluating quality of sequence reads.
- `*_cds_from_genomic.fna.gz` coding sequences (fasta) which have been annotated. 
- `*_feature_count.txt.gz` count of features in `*_feature_table.txt.gz`. 
- `*_feature_table.txt.gz` 
- **`*_genomic.fna` the actual sequence data (fasta). Sequences are identified by GenBank accession. Sequences do NOT follow the order listed in `*_assembly_report.txt`.**
- **`*_genomic_gaps.txt` lists all start and end coordinates of sequences of `N` longer than 10 base pairs. Sequences are identified by GenBank accession. Use this to un-gap sequence data. File includes single-line header starting with `#`. Columns are:**

    0. **accession.version**: accession identifier.
    1. **start**: start coordinate. All coordinates are the number of base pairs from the start of the sequence from `*_genomic.fna`. Note this is NOT zero-indexed (i.e. 1-10000 has length 10000): subtract 1 to index correctly in python.
    2. **stop**: end coordinate.
    3. gap_length: length in base pairs.
    4. gap_type: classifies gap (e.g. `telomere`)
    5. linkage_evidence: not sure what this means.

- `*_genomic.gbff` GenBank Flat File combines sequence and all annotations (including gaps). Contains same information as other files, but not as machine readable. 
- `*_genomic.gff` general feature format file (tab delimited table) listing annotations. Includes centromere start/end positions, may be relevant.
- `*_genomic.gtf` gene transfer format file. 
- `*_protein.faa.gz` annotated protein sequences (fasta). 
- `*_protein.gpff.gz` 
- `*_rm.out.gz` 
- `*_rm.run` 
- `*_rna.fna.gz` annotated RNA sequences (fasta). 
- `*_rna.gbff.gz file`
- `*_rna_from_genomic.fna.gz` annotated RNA sequences (fasta). 
- `*_translated_cds.faa.gz`
- `*_wgsmaster.gbff.gz`
       GenBank flat file format of the WGS master for the assembly (present only
       if a WGS master record exists for the sequences in the assembly).
- `annotation_hashes.txt` checksums for specific columns of the annotation data, to check for changes to names or features. 
- **`md5checksums.txt` md5 checksums for all files in directory. Hard code these values to check download integrity of files. File has no header. Columns are:**

    0. md5 checksum.
    1. relative path to file from root directory of assembly.
