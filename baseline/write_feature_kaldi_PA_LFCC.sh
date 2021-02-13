#!/bin/bash

matlab -nodisplay -nosplash <<EOF
maxNumCompThreads(1);
write_feature_kaldi_PA_LFCC_train
write_feature_kaldi_PA_LFCC_dev
write_feature_kaldi_PA_LFCC_eval
EOF

# convert .txt file into kaldi formats (.ark and .scp)
for name in PA_LFCC_train PA_LFCC_dev PA_LFCC_eval; do
        copy-feats ark,t:data/lfcc/${name}.txt ark,scp:`pwd`/data/lfcc/${name}/feats.ark,`pwd`/data/lfcc/${name}/feats.scp
done

