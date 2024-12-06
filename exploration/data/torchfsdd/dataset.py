import os, random, glob, torch, torchaudio

class TorchFSDDGenerator:
    """A :class:`torch:torch.utils.data.Dataset` generator for splits of the Free Spoken Digit Dataset.

    Parameters
    ----------
    version: str
        The version of FSDD to download from `the GitHub repository <https://github.com/Jakobovski/free-spoken-digit-dataset>`_,
        specified as a branch name (defaults to `'master'`) or Git version tag, e.g. `'v1.0.6'`.

        Alternatively, if you already have a local copy of the dataset that you would like to use,
        you can set this argument to `'local'` and provide a path to the folder containing the WAV files, as the ``path`` argument.

    path: str, optional
        If ``version`` is a Git branch name or version tag, then this is the path where the Git repository will be cloned to
        (a new folder will be created at the specified path). If none is specified, then :py:func:`python:os.getcwd` is used.

        If ``version`` is set to `'local'`, then this is the path to the folder containing the WAV audio recordings.

    transforms: callable, optional
        A callable transformation to apply to a 1D :class:`torch:torch.Tensor` of audio samples.

        .. seealso::

            :class:`TorchFSDD`

    load_all: bool
        Whether or not to load the entire dataset into memory.

        .. seealso::

            :class:`TorchFSDD`

    **args: optional
        Arbitrary keyword arguments passed on to :py:func:`torchaudio:torchaudio.load`.
    """
    def __init__(self, version='local', path=None, train_transforms=None, val_transforms = None,  **args):
        if version == 'local':
            if path is None:
                raise ValueError('Expected path to be a directory containing WAV recordings')
        else:
            assert False, "only local files supported"

        self.path = path
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.args = args
        self.all_files = glob.glob(self.path + '/*.wav')

    def full(self):
        """Generates a data set wrapper for the entire data set.

        Returns
        -------
        full_set: :class:`TorchFSDD`
            The :class:`torch:torch.utils.data.Dataset` wrapper for the full data set.
        """
        return TorchFSDD(self.all_files, self.transforms, self.load_all, **self.args)

 
    def train_val_test_split(self, test_size=0.1, val_size=0.1):
        """Generates training, validation and test data set wrappers.

        Parameters
        ----------
        test_size: 0 < float < 1
            Size of the test data set (as a proportion).

        val_size: 0 < float < 1
            Size of the validation data set (as a proportion).

        Returns
        -------
        train_set: :class:`TorchFSDD`
            The training set :class:`torch:torch.utils.data.Dataset` wrapper.

        val_set: :class:`TorchFSDD`
            The validation set :class:`torch:torch.utils.data.Dataset` wrapper.

        test_set: :class:`TorchFSDD`
            The test set :class:`torch:torch.utils.data.Dataset` wrapper.
        """
        assert 0. < test_size < 1.
        assert 0. < val_size < 1.
        assert test_size + val_size < 1.

        n_rec = len(self.all_files)
        # print(n_rec)
        n_test, n_val = int(n_rec * test_size), int(n_rec * val_size)

        random.shuffle(self.all_files)

        val_files = self.all_files[:n_val]
        test_files = self.all_files[n_val:(n_val + n_test)]
        train_files = self.all_files[(n_val + n_test):]

        train_set = TorchFSDD(train_files, self.train_transforms, **self.args)
        val_set = TorchFSDD(val_files, self.val_transforms , **self.args)
        test_set = TorchFSDD(test_files, self.val_transforms, **self.args)
        return train_set, val_set, test_set

WAV_LEN = 8192
class TorchFSDD(torch.utils.data.Dataset):
    """A :class:`torch:torch.utils.data.Dataset` wrapper for specified
    WAV audio recordings of the Free Spoken Digit Dataset.

    .. tip::

        There should rarely be a situation where you have to initialize this class manually,
        unless you are experimenting with specific subsets of the FSDD. You should use :class:`TorchFSDDGenerator`
        to either load the full data set or generate splits for training/validation/testing.

    Parameters
    ----------
    files: list of str
        List of file paths to the WAV audio recordings for the dataset.

    transforms: callable, optional
        A callable transformation to apply to a 1D :class:`torch:torch.Tensor` of audio samples.

        This can be a single transformation, such as the :class:`TrimSilence` transformation included in this package.

        .. code-block:: python

            from torchfsdd import TorchFSDDGenerator, TrimSilence

            fsdd = TorchFSDDGenerator(transforms=TrimSilence(threshold=150))

        It could also be a series of transformations composed together with :class:`torchvision:torchvision.transforms.Compose`.

        .. code-block:: python

            from torchfsdd import TorchFSDDGenerator, TrimSilence
            from torchaudio.transforms import MFCC
            from torchvision.transforms import Compose

            fsdd = TorchFSDDGenerator(transforms=Compose([
                TrimSilence(threshold=100),
                MFCC(sample_rate=8e3, n_mfcc=13)
            ]))

        There are many useful audio transformations in :py:mod:`torchaudio:torchaudio.transforms` such as :class:`torchaudio:torchaudio.transforms.MFCC`.

    load_all: bool
        Whether or not to load the entire dataset into memory.

        This essentially defeats the point of batching, but the dataset is relatively small
        enough that it can comfortably fit into memory and possibly provide some speed-up.

        If this is set to `True`, then the complete set of raw audio recordings and labels
        (for the specified files) can be accessed with ``self.recordings`` and ``self.labels``.

    **args: optional
        Arbitrary keyword arguments passed on to :py:func:`torchaudio:torchaudio.load`.
    """
    def __init__(self, files, transforms=None, cache_transformed = True, **args):
        super().__init__()
        self.all_files = files
        self.files = []
        self.transforms = transforms
        self.args = args
        self.do_cache = cache_transformed

        get_audio = lambda file: torchaudio.load(file, **self.args)[0]
        get_label = lambda file: int(os.path.basename(file)[0])

        self.recordings, self.labels = [], []
        for file in self.all_files:
            audio = get_audio(file)
            if audio.numel() < 8001: # reject files with unnessarily long waveforms
                audio = torch.nn.functional.pad(audio, (0, WAV_LEN - audio.numel()), "constant", 0.0)
                self.recordings.append(audio)
                self.labels.append(get_label(file))
                self.files.append(file)
        self.transformed_recording_cache = {}

        def _load(self, index):
            return self.recordings[index], self.labels[index]

        setattr(self.__class__, '_load', _load)

    def __len__(self):
        return len(self.recordings)

    #FIXME: this dataset is under suspicion of containing a zero or nan wave file which is fucking shit up
    def __getitem__(self, index):
        if index in self.transformed_recording_cache.keys():
            return self.transformed_recording_cache[index], self.labels[index]

        # Fetch the audio and corresponding label
        x, y = self._load(index)
        x = x.flatten()

        # Transform data if a transformation is given
        if self.transforms is not None:
            x = self.transforms(x.reshape(1,1,-1))
            if isinstance(x, dict):
                x = x['samples']
            self.transformed_recording_cache[index] = x

        return x, y