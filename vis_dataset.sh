# PATH=/usr/local/bin:$PATH xvfb-run -s "-screen 0 1024x768x24" python visualize/visualize_sequences.py --file_path save/diffh2o_full/samples_000200000/results_test_objects_unseen.npy  --is_pca --save_video --resolution low

### preprocess oakink2 data
python -m prepare.preprocess_oakink2 --mode both

#   xvfb-run -s "-screen 0 1024x768x24" python visualize/visualize_oakink2.py \
#       --motion_path dataset/OAKINK2/oakink2_primitive/000000.npy \
#       --output_path output/oakink2_vis_merged.mp4 \
#       --max_frames 2000



#  xvfb-run -s "-screen 0 1024x768x24" python visualize/visualize_oakink2.py \
#       --raw_anno /hhd4/lizhe/dataset/OakInk2/data/anno_preview/scene_01__A001++seq__6e11d637b13834d44e77__2023-04-27-19-22-25.pkl \
#       --output_path output/raw_aligned_object_hand_side.mp4 \
#       --resolution low \
#       --view side \
#       --render_mode skelton

#  xvfb-run -e /tmp/xvfb_101.err -a -s "-screen 0 1024x768x24" python visualize/visualize_oakink2.py \
#       --raw_anno /hhd4/lizhe/dataset/OakInk2/data/anno_preview/scene_01__A001++seq__6e11d637b13834d44e77__2023-04-27-19-22-25.pkl \
#       --output_path output/raw_aligned_object_hand_side.mp4 \
#       --resolution low \
#       --view side \
#       --render_mode skeleton