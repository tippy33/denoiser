# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

from collections import namedtuple
import json
from pathlib import Path
import math
import os
import sys

import torchaudio
from torch.nn import functional as F

from .dsp import convert_audio
import soundfile as sf
import torch
import torch.nn.functional as F

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    info = sf.info(path)
    return Info(
        length=info.frames, 
        sample_rate=info.samplerate, 
        channels=info.channels
    )


def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta

def find_audio_file(path, exts=[".wav"]):
    meta = []
    info = get_info(path)
    meta.append((path, info.length))
    return meta


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None,
                 channels=None, convert=False):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.convert = convert
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            data, sr = sf.read(str(file), always_2d=True) # (frames, channels)
            data = data.T # (channels, frames)
            
            if num_frames > 0:
                data = data[:, offset : offset + num_frames]
            elif offset > 0:
                data = data[:, offset:]
                
            out = torch.from_numpy(data).float()

            target_sr = self.sample_rate or sr
            target_channels = self.channels or out.shape[0]
            
            if self.convert:
                # convert_audio function needs to be defined
                out = convert_audio(out, sr, target_sr, target_channels)
            else:
                if sr != target_sr:
                    raise RuntimeError(f"Expected {file} to have sample rate of {target_sr}, but got {sr}")
                if out.shape[0] != target_channels:
                    raise RuntimeError(f"Expected {file} to have channels of {target_channels}, but got {out.shape[0]}")
            
            if num_frames and num_frames > out.shape[-1]:
                out = F.pad(out, (0, num_frames - out.shape[-1]))

            if self.with_path:
                return out, str(file)
            else:
                return out


if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_audio_files(path)
    json.dump(meta, sys.stdout, indent=4)
