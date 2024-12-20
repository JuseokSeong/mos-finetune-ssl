# ==============================================================================
# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

## Get a pretrained model and run inference given a directory of .wav files;
## generate an answers.txt file containing predictions per utterance.

import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your directory of wav files')
    parser.add_argument('--outfile', type=str, required=False, default='answer.txt', help='Output filename of the result')
    args = parser.parse_args()
    DATADIR = args.datadir

    ## 1. download the base model from fairseq
    if not os.path.exists('fairseq/wav2vec_small.pt'):
        os.system('mkdir -p fairseq')
        os.system('wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -P fairseq')
        os.system('wget https://raw.githubusercontent.com/pytorch/fairseq/main/LICENSE -P fairseq/')

    ## 2. download the finetuned checkpoint
    if not os.path.exists('pretrained/ckpt_w2vsmall'):
        os.system('mkdir -p pretrained')
        os.system('wget https://zenodo.org/record/6785056/files/ckpt_w2vsmall.tar.gz')
        os.system('tar -zxvf ckpt_w2vsmall.tar.gz')
        os.system('mv ckpt_w2vsmall pretrained/')
        os.system('rm ckpt_w2vsmall.tar.gz')
        os.system('cp fairseq/LICENSE pretrained/')

    ## 3. run inference
    os.system('python predict_noGT.py --fairseq_base_model fairseq/wav2vec_small.pt --outfile answer.txt --finetuned_checkpoint pretrained/ckpt_w2vsmall --datadir ' + DATADIR + ' --outfile ' + args.outfile)

main()
