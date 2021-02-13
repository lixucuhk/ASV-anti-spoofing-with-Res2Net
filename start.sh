
mkdir -p logs || exit 1

echo "Start training."
python3 train.py --run-id SEResNet34PASpec0 --random-seed 0 --data-feats pa_spec --configfile conf/training_mdl/seresnet34.json >logs/SEResNet34PASpec0 || exit 1

echo "Start evaluation on all checkpoints."
for model in model_snapshots/SEResNet34PASpec0/*_[0-9]*.pth.tar; do
    python3 eval.py --random-seed 0 --data-feats pa_spec --configfile conf/training_mdl/seresnet34.json --pretrained $model || exit 1
done

