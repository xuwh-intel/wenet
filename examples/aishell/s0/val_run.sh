#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

stage=5 # start from 0 if you need to start from data preparation
stop_stage=5


# The aishell dataset location, please change this to your own path
# make sure of using absolute path. DO-NOT-USE relatvie path!
data=/ssd/wenhaoxu/datasets/asr-data/OpenSLR/33/
data_url=www.openslr.org/resources/33

dict=data/dict/lang_char.txt

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_set=train
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_u2++_conformer.yaml: U2++ conformer
# 6. conf/train_u2++_transformer.yaml: U2++ transformer
train_config=conf/train_conformer.yaml
cmvn=true
dir=exp/conformer
checkpoint=
num_workers=8
prefetch=500

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=3
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  dir=exp/wenet_efficient_conformer_aishell_v2
  decode_modes="ctc_greedy_search attention_rescoring"
  ctc_weight=0.5
  reverse_weight=0.3
  decoding_chunk_size=-1
  python wenet/bin/recognize.py --gpu 0 \
    --modes $decode_modes \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data data/test/data.list \
    --checkpoint $dir/final.pt \
    --beam_size 10 \
    --batch_size 1 \
    --penalty 0.0 \
    --dict $dir/words.txt \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $dir \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} > logs/decode_aishell.log
  for mode in ${decode_modes}; do
    python tools/compute-cer.py --char=1 --v=1 \
      data/test/text $dir/${mode}/text > $dir/${mode}/cer.txt
  done
fi
