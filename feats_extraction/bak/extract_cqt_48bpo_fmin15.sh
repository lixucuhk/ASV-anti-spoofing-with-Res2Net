##
python3 local/preprocess/compute_CQT.py --out_dir ./PA_LPCQT_16msHop48BPOfmin15 --access_type PA --param_json_path ./conf/cqt_48bpo_fmin15.json || exit 1
python3 GenLPCQTFeats_kaldi.py --access_type PA --work_dir PA_LPCQT_16msHop48BPOfmin15 || exit 1

