class CFG:
    data_p = f"/content/data"
    main_df_path = f'{data_p}/train.csv'
    train_df_path = f'{data_p}/new_train.csv'
    valid_df_path = f'{data_p}/new_valid.csv'
    train_features_path = f'{data_p}/Features_train.pkl'
    valid_features_path = f'{data_p}/Features_valid.pkl'
    video_path = f'{data_p}/videos_train'
    out_dir = f'/content/drive/MyDrive/Olimpiads/nto_hack_2022/V2'

    model_name = 'v2_1'
    backbone = 'gpt2'
    prefix_length = 35
    
    epochs = 27
    save_every = 1
    batch_size = 30
    learning_rate = 5e-5
    warmup_steps = 5000