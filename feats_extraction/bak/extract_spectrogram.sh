#!/bin/bash
# extract logspec features for:
# ASVspoof2019 LA train, LA dev, LA eval, PA train, PA dev, PA eval

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
fbankdir=`pwd`/fbank
specdir=`pwd`/logspec
vadir=`pwd`/mfcc

stage=0

. ./parse_options.sh || exit 1


echo 'stage is '$stage
if [ $stage -eq 0 ]; then
	# first create spk2utt 
	for name in train dev eval; do

		[ -d data/${name}_spec ] || cp -r data/${name} data/${name}_spec || exit 1
		local/make_spectrogram.sh --spectrogram-config conf/spec.conf --nj 80 --cmd "$train_cmd" \
		  data/${name}_spec exp/make_spec $specdir
	
      done 
fi 

