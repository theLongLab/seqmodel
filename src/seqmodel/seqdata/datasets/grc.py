import os

# from https://github.com/pytorch/vision/blob/master/torchvision/datasets

class SequenceDataset():

    def __init__(self):
        self.root = ""  # TODO

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file))



class ReferenceGenome(SequenceDataset):

    resources = {  # data type
        "full_analysis_set":
        {
            # source: (url, md5)
            "GRCh38.p13": [("ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.28_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_full_analysis_set.fna.gz", "a03308ba679819b15442f459777c613c")],
        }
    }

    def __init__(
            self,
            root: str,
            download: bool = False,
            data_type: str = "full_analysis_set",
            source: str = "GRCh38.p13",
            ):
            self.data_type = data_type
            self.source = source
            if download:
                self.download()
        pass

    def download(self, source, version):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)


    @property
    def raw_folder(self, source, version) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')