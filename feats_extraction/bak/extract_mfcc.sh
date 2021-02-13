#!/bin/bash
# extract fbank, mfcc, logspec, ivector features for:
# ASVspoof2019 LA train, LA dev, LA eval, PA train, PA dev, PA eval

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
fbankdir=`pwd`/fbank
specdir=`pwd`/logspec
vaddir=`pwd`/mfcc

stage=-1

. ./parse_options.sh || exit 1


echo 'stage is '$stage


if [ $stage -eq -1 ]; then
	# first create spk2utt 
	for name in pa_train pa_dev pa_eval la_train la_dev la_eval; do
		[ -d data/${name}_mfcc ] ||  cp -r data/$name data/${name}_mfcc || exit 1
		[ -f data/${name}_mfcc/wav.scp.new ] || python3 MakeSpkidPrefixed.py data/${name}_mfcc/utt2spk data/${name}_mfcc/wav.scp || exit 1
		cp data/${name}_mfcc/wav.scp.new data/${name}_mfcc/wav.scp || exit 1
		cp data/${name}_mfcc/utt2spk.new data/${name}_mfcc/utt2spk || exit 1
		utils/utt2spk_to_spk2utt.pl data/${name}/utt2spk > data/${name}/spk2utt
                utils/fix_data_dir.sh  data/${name}_mfcc || exit 1

		steps/make_mfcc.sh  --mfcc-config conf/mfcc.conf --nj 20 --cmd "$train_cmd" \
                                                 data/${name}_mfcc exp/make_mfcc $mfccdir || exit 1
		utils/fix_data_dir.sh  data/${name}_mfcc || exit 1
		sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" data/${name}_mfcc exp/make_vad $vaddir || exit 1
		utils/fix_data_dir.sh data/${name}_mfcc || exit 1
        done
fi


