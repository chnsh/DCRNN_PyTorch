Quick Run:(All based on METR-LA)
1. pip install -r requirements.txt
2. Download the data from https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX
3. Put the data into data/ for making the training data
4. Create data directories
>> mkdir -p data/{METR-LA,PEMS-BAY}
5. generate train/test/val dataset at data/{METR-LA,PEMS-BAY}/{train,val,test}.npz
>> python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
6. Constructing the Graph
>> python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
    --output_pkl_filename=data/sensor_graph/adj_mx.pkl
7. Run the pre-trained model:
>> python run_demo_pytorch.py --config_filename=data/model/pretrained/METR-LA/config.yaml
8. Model Training 
>> python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml
9. Evaluating the baselines:
>> python -m scripts.eval_baseline_methods --traffic_reading_filename=data/metr-la.h5