python main.py \
  --dir_data /home/liux/datasets/nyudepth_hdf5 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --split_json ./data/nyu.json --num_sample 500 \
