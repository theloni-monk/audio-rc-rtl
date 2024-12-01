
import torch
from torch import Tensor
from typing import Optional

import numpy as np
import torchaudio as ta
import random
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.mel_scale import convert_frequencies_to_mels, convert_mels_to_frequencies
from torch_audiomentations.utils.object_dict import ObjectDict

class AllPassFilter(BaseWaveformTransform):
    def __init__(self,  min_center_frequency=100,
                        max_center_frequency=3500,
                        min_Q=0.1,
                        max_Q=1.0,
                        mode: str = "per_example",
                        p: float = 0.1,
                        p_mode: str = None,
                        sample_rate: int = None,
                        target_rate: int = None,
                        output_type = None):
        super().__init__(mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type)
        self.min_center_frequency = min_center_frequency
        self.max_center_frequency = max_center_frequency
        self.min_Q = min_Q
        self.max_Q = max_Q

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        :params samples: (batch_size, num_channels, num_samples)
        """

        batch_size, _, num_samples = samples.shape

        # Sample frequencies uniformly in mel space, then convert back to frequency
        def get_dist(min_freq, max_freq):
            dist = torch.distributions.Uniform(
                low=convert_frequencies_to_mels(
                    torch.tensor(min_freq, dtype=torch.float32, device=samples.device)
                ),
                high=convert_frequencies_to_mels(
                    torch.tensor(max_freq, dtype=torch.float32, device=samples.device)
                ),
                validate_args=True,
            )
            return dist

        center_dist = get_dist(self.min_center_frequency, self.max_center_frequency)
        self.transform_parameters["center_freq"] = convert_mels_to_frequencies(
            center_dist.sample(sample_shape=(batch_size,))
        )

        Q_dist = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_Q, dtype=torch.float32, device=samples.device
            ),
            high=torch.tensor(
                self.max_Q, dtype=torch.float32, device=samples.device
            ),
        )
        self.transform_parameters["Q"] = Q_dist.sample(
            sample_shape=(batch_size,)
        )
    
    def apply_transform(self, 
                        samples,
                        sample_rate,
                        targets,
                        target_rate):
        
        freq = self.transform_parameters["center_freq"]
        Q = self.transform_parameters["Q"]

        samps = ta.functional.allpass_biquad(samples, self.sample_rate if sample_rate is None else sample_rate, freq, Q)

        return ObjectDict(
            samples=samps,
            sample_rate = sample_rate,
            targets = targets,
            target_rate = target_rate
        )