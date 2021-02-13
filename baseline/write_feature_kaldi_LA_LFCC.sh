#!/bin/bash

matlab -nodisplay -nosplash <<EOF
maxNumCompThreads(1);
write_feature_kaldi_LA_LFCC_train
write_feature_kaldi_LA_LFCC_dev
write_feature_kaldi_LA_LFCC_eval
EOF

# convert .txt file into kaldi formats (.ark and .scp)
for name in LA_LFCC_train LA_LFCC_dev LA_LFCC_eval; do
        copy-feats ark,t:data/lfcc/${name}.txt ark,scp:`pwd`/data/lfcc/${name}/feats.ark,`pwd`/data/lfcc/${name}/feats.scp
done

