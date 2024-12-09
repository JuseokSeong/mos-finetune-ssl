# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

## run inference without requiring ground-truth answers
## or system info.

import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from mos_fairseq import MosPredictor #, MyDataset
import numpy as np
import scipy.stats
import datetime
import time

import glob
import librosa

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def unixnow():
    return str(int(time.mktime(datetime.datetime.now().timetuple())))


def systemID(uttID):
    return uttID.split('-')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model.')
    parser.add_argument('--datadir', type=str, required=True, help='Path of your directory containing .wav files')
    parser.add_argument('--finetuned_checkpoint', type=str, required=True, help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--outfile', type=str, required=False, default='answer.txt', help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
    args = parser.parse_args()
    
    cp_path = args.fairseq_base_model
    my_checkpoint = args.finetuned_checkpoint
    wavdir = args.datadir
    outfile = args.outfile

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()
    model.load_state_dict(torch.load(my_checkpoint, map_location=device))
    
    ans = open(outfile, 'w')
    with open(wavdir, 'r') as f:
        for l in f:
            items = l.strip().split()
            wav_id = items[0]
            fname = items[1]
            waveform = librosa.load(fname, sr=16000)[0]

            # inputs, labels, filenames = data
            inputs = torch.tensor(waveform).unsqueeze(0).to(device)
            outputs = model(inputs)
            
            output = outputs.cpu().detach().numpy()[0]
            ans.write(wav_id + " " + str(output) + "\n")
    ans.close()

if __name__ == '__main__':
    main()
