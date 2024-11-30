
import torch
import numpy as np
import torchaudio as ta
import random

def RandomAllPass(waveform):
    return ta.functional.allpass_biquad(waveform, 8000, random.random()*3500, random.random())
