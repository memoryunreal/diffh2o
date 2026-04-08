# $modelpath="./save/diffh2o_full/model000200000.pt"
# modelpath="./save/diffh2o_interaction/model000200000.pt"
modelpath="./save/oakink2_full/model000200000.pt"
# $filepath="save/diffh2o_full/samples_000200000/results_test_objects_unseen.npy"
filepath="save/diffh2o_interaction/samples_000200000/results_test_objects_unseen.npy"

python -m sample.generate --model_path $modelpath --num_samples 1

# xvfb-run -a python visualize/visualize_sequences.py --file_path $filepath --is_pca --save_video --resolution low

# python visualize/visualize_sequences.py --file_path save/diffh2o_full/samples_000400000/ --is_pca --save_video
