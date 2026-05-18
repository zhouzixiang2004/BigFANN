#!/usr/bin/env bash

set -euo pipefail

# https://github.com/KGLab-HDU/TKDE-under-review-Native-Hybrid-Queries-via-ANNS?tab=readme-ov-file
pip3 install gdown
cd data
# if [ ! -d sift ]; then
#     wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
#     tar -xf sift.tar.gz
# fi
# if [ ! -d sift_label ]; then
#     python -m gdown https://drive.google.com/uc?id=15sflYLREoqHJGJCuBpiE1UOHad60_GKK
#     tar -xf sift_label.tar.gz
# fi
# if [ ! -f sift/label_sift_groundtruth.ivecs ]; then
#     cd sift
#     python -m gdown https://drive.google.com/uc?id=1MVw1QmhQ_TnfhAV3Np-PDH9GNnH3Vm0w
#     cd ..
# fi
if [ ! -d gist ]; then
    wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
    tar -xf gist.tar.gz
fi
if [ ! -d gist_label ]; then
    python -m gdown https://drive.google.com/uc?id=1PFeQev-7jywvdOVXy5ubMhltbH5sFDRx
    tar -xf gist_label.tar.gz
fi
if [ ! -f gist/label_gist_groundtruth.ivecs ]; then
    cd gist
    python -m gdown https://drive.google.com/uc?id=1KkeEbEglX6plVy4rT4GKkhCTKnOQ9jbh
    cd ..
fi
# if [ ! -d glove-100 ]; then
#     wget http://downloads.zjulearning.org.cn/data/glove-100.tar.gz
#     tar -xf glove-100.tar.gz
# fi
# if [ ! -d glove-100_label ]; then
#     python -m gdown https://drive.google.com/uc?id=10bIhmw1RC4Bk6cpJuWRli1WuwbALEKuK
#     tar -xf glove-100_label.tar.gz
# fi
# if [ ! -f glove-100/label_glove-100_groundtruth.ivecs ]; then
#     cd glove-100
#     python -m gdown https://drive.google.com/uc?id=1LHbXi6Aapvnxp68aGZF1DV3kXy23bFE_
#     cd ..
# fi
# if [ ! -d crawl ]; then
#     wget http://downloads.zjulearning.org.cn/data/crawl.tar.gz
#     tar -xf crawl.tar.gz
# fi
# if [ ! -d crawl_label ]; then
#     python -m gdown https://drive.google.com/uc?id=1d1TURrWxYAELvfiBNermEv0iiyTxAWF6
#     tar -xf crawl_label.tar.gz
# fi
# if [ ! -d audio ]; then
#     python -m gdown https://drive.google.com/uc?id=1fJvLMXZ8_rTrnzivvOXiy_iP91vDyQhs
#     tar -xf audio.tar.gz
# fi
# if [ ! -d audio_label ]; then
#     python -m gdown https://drive.google.com/uc?id=1IsAGjhDSu2xrh2w16iVBEfw9vbOCRYjq
#     tar -xf audio_label.tar.gz
# fi
# if [ ! -f audio/label_audio_groundtruth.ivecs ]; then
#     cd audio
#     python -m gdown https://drive.google.com/uc?id=1WeBC4_Aw2pfM_DlFaJUuM0GRuLAPCI3P
#     cd ..
# fi
# if [ ! -d msong ]; then
#     python -m gdown https://drive.google.com/uc?id=1UZ0T-nio8i2V8HetAx4-kt_FMK-GphHj
#     tar -xf msong.tar.gz
# fi
# if [ ! -d msong_label ]; then
#     python -m gdown https://drive.google.com/uc?id=1jVpJaT5GRjxRzj4C3KSsev0clQIOEplZ
#     tar -xf msong_label.tar.gz
# fi
# if [ ! -f msong/label_msong_groundtruth.ivecs ]; then
#     cd msong
#     python -m gdown https://drive.google.com/uc?id=1LFWshAIoQLYJx68toTQBaoIOBZDfExue
#     cd ..
# fi
# if [ ! -d enron ]; then
#     python -m gdown https://drive.google.com/uc?id=1TqV43kzuNYgAYXvXTKsAG1-ZKtcaYsmr
#     tar -xf enron.tar.gz
# fi
# if [ ! -d enron_label ]; then
#     python -m gdown https://drive.google.com/uc?id=1tbVjQlUlFS321CxW9_hfqUf4JUiXdmLi
#     tar -xf enron_label.tar.gz
# fi
# if [ ! -f enron/label_enron_groundtruth.ivecs ]; then
#     cd enron
#     python -m gdown https://drive.google.com/uc?id=1F5eZwG_u8S3StwPOnlmrHqmoFCoaGKVB
#     cd ..
# fi
# if [ ! -d uqv ]; then
#     python -m gdown https://drive.google.com/uc?id=1HIdQSKGh7cfC7TnRvrA2dnkHBNkVHGsF
#     tar -xf uqv.tar.gz
# fi
# if [ ! -d uqv_label ]; then
#     python -m gdown https://drive.google.com/uc?id=1YN6VuLPw_u9cFREXS6jgApYjCTmzmZtv
#     tar -xf uqv_label.tar.gz
# fi
# if [ ! -f uqv/label_uqv_groundtruth.ivecs ]; then
#     cd uqv
#     python -m gdown https://drive.google.com/uc?id=1o05Iq9Q_omnHosWnrwRQBYXtN4n7nu5o
#     cd ..
# fi
if [ ! -d paper ]; then
    python -m gdown https://drive.google.com/uc?id=1t4b93_1Viuudzd5D3I6_9_9Guwm1vmTn
    tar -xf paper.tar.gz
fi
if [ ! -d paper_label ]; then
    python -m gdown https://drive.google.com/uc?id=1arpB0oZne3tmRCUfTfzQmIfvWVP_kuKY
    tar -xf paper_label.tar.gz
fi
