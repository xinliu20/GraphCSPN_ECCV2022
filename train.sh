python main.py \
  --dir_data /path/to/your/dataset \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --split_json ./data/nyu.json --num_sample 500 \
