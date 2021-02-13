
mkdir -p logs || exit 1

randomseed=0
config=conf/training_mdl/seresnet34.json
feats=debug_feats
runid=SEResNet34Debugfeats0

# echo "Start training."
# python3 train.py --run-id $runid --random-seed $randomseed --data-feats $feats --configfile $config >logs/$runid || exit 1

echo "Start evaluation on all checkpoints."
for model in model_snapshots/$runid/*_[0-9]*.pth.tar; do
    python3 eval.py --random-seed $randomseed --data-feats $feats --configfile $config --pretrained $model || exit 1
done

