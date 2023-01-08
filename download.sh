#!/bin/bash

DATA_ROOT=./voxforge/enlang
DATA_MAIN=./data

DATA_SRC="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit" 
DATA_TGZ=${DATA_ROOT}/tgz
DATA_EXTRACT=${DATA_ROOT}/extracted

mkdir -p ${DATA_TGZ} 2>/dev/null

# Check if the executables needed for this script are present in the system
command -v wget >/dev/null 2>&1 ||\
    { echo "\"wget\" is needed but not found"'!'; exit 1; }

echo "--- Starting VoxForge data download (may take some time) ..."
wget -P ${DATA_TGZ} -l 1 -N -nd -c -e robots=off -A tgz -r -np ${DATA_SRC} || \
    { echo "WGET error"'!' ; exit 1 ; }

 