# Commonly used funcs.
import pandas as pd
import torch
import numpy as np
import random
import os
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler


def setup_seed(seed):
    """
    Set seed
    Args:
        seed:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

def save_data_idx(dataset, arr_flag=False, arr_path=None):
    """
    因为objective function会被执行很多次，所以这里先保存一下idx，使得所有objective在同一组dataset上进行。
    然后在tuning时，会在shuffle_time组不同的dataset split上进行，用以取平均
    Args:
        dataset:
        opt:
        shuffle_time: The num of list of index to be generated.
    """
    shuffled_indices = []

    # 使用全部的data
    if not arr_flag:
        DATA_len = dataset.__len__()
        shuffled_indices = np.random.permutation(DATA_len)

    # 使用指定的data
    else:
        shuffled_indices = np.load(arr_path)
        # DATA_len = len(shuffled_indices)
        np.random.shuffle(shuffled_indices)

    return shuffled_indices

def save_data_idx_simplified(LEN):
    """
    因为objective function会被执行很多次，所以这里先保存一下idx，使得所有objective在同一组dataset上进行。
    然后在tuning时，会在shuffle_time组不同的dataset split上进行，用以取平均
    Args:
    dataset:
    opt:
    shuffle_time: The num of list of index to be generated.
    """
    shuffled_indices = np.random.permutation(LEN)

    return shuffled_indices

def get_data_idx(shuffled_indices,train_pct, vali_pct):
    """
    To get data split idx according to shuffled 'shuffled_indices'
    """

    DATA_len = len(shuffled_indices)

    train_idx = shuffled_indices[:int(train_pct * DATA_len)]

    # 20% for testing
    test_pct = 1-train_pct-vali_pct

    tmp = int((train_pct + test_pct) * DATA_len)
    test_idx = shuffled_indices[int(train_pct * DATA_len):tmp]
    val_idx = shuffled_indices[tmp:]  # 10 % for validation
    return train_idx,val_idx,test_idx

def get_data_loader(dataset, batch_size, train_idx,val_idx,test_idx , collate_fn):
    """
    To get dataloader according to shuffled 'shuffled_indices'
    Args:
        dataset:
        shuffled_indices:
        opt:

    Returns:

    """

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(train_idx), collate_fn=collate_fn)
    val_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx), collate_fn=collate_fn)
    # 注意test_loader的batch size
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(test_idx), collate_fn=collate_fn)

    return train_loader,val_loader,test_loader

def get_data_loader_er(dataset, batch_size, train_idx,val_idx,test_idx , collate_fn):

    # 注意test_loader的batch size
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                            sampler=SubsetRandomSampler(test_idx), collate_fn=collate_fn)
    return test_loader

def get_data_idx_bidfee(shuffled_indices, opt, FEE=0.01):
    """
    To get data split idx according to shuffled 'shuffled_indices'
    Args:
        shuffled_indices:
        FEE: bid fee selected as non-training set
        opt:

    Returns:
    """
    data_key = pd.read_csv(opt.data_key_path)

    non_test_idx = [i for i in range(0,len(data_key)) if data_key.loc[i,'bidfee'] != FEE]

    test_idx = [i for i in range(len(data_key)) if i not in non_test_idx]
    print(f"testing set size = {len(test_idx)}")

    tmp = int((opt.vali_pct) * len(non_test_idx))
    val_idx = non_test_idx[-tmp:]
    train_idx = [i for i in non_test_idx if i not in val_idx]

    del data_key
    return train_idx,val_idx,test_idx

def get_data_idx_bidinc(shuffled_indices,opt,INC=0.01):
    """
    To get data split idx according to shuffled 'shuffled_indices'
    Args:
        shuffled_indices:
        FEE: bid fee selected as non-training set
        opt:

    Returns:
    """
    data_key = pd.read_csv(opt.data_key_path)

    non_test_idx = [i for i in range(0,len(data_key)) if data_key.loc[i,'bidincrement'] != INC]

    test_idx = [i for i in range(len(data_key)) if i not in non_test_idx]
    print(f"testing set size = {len(test_idx)}")

    # Vali set is 10%
    tmp = int((opt.vali_pct) * len(non_test_idx))
    val_idx = non_test_idx[-tmp:]
    train_idx = [i for i in non_test_idx if i not in val_idx]


    del data_key
    return train_idx,val_idx,test_idx

def get_data_idx_retail(shuffled_indices, opt, RETAIL=292.495):
    """
    To get data split idx according to shuffled 'shuffled_indices'
    Args:
        shuffled_indices:
        FEE: bid fee selected as non-training set
        opt:

    Returns:
    """
    data_key = pd.read_csv(opt.data_key_path)

    non_test_idx = [i for i in range(0,len(data_key)) if data_key.loc[i,'retail'] <= RETAIL]

    test_idx = [i for i in range(len(data_key)) if i not in non_test_idx]
    print(f"testing set size = {len(test_idx)}")

    tmp = int((opt.vali_pct) * len(non_test_idx))
    val_idx = non_test_idx[-tmp:]
    train_idx = [i for i in non_test_idx if i not in val_idx]

    del data_key
    return train_idx,val_idx,test_idx

def get_data_idx_emb(shuffled_indices, opt):
    """
    To get data split idx according to shuffled 'shuffled_indices'
    Args:
        shuffled_indices:
        opt:

    Returns:
    """
    cluster_assign = pd.read_csv(opt.cluster_assign_path)

    non_test_idx = [i for i in range(len(cluster_assign)) if cluster_assign.loc[i, 'cluster'] != opt.CHOSEN_CLUSTER]

    test_idx = [i for i in range(len(cluster_assign)) if i not in non_test_idx]
    print(f"testing set size = {len(test_idx)}")


    tmp = int((opt.vali_pct) * len(non_test_idx))
    val_idx = non_test_idx[-tmp:]
    train_idx = [i for i in non_test_idx if i not in val_idx]

    del cluster_assign
    return train_idx, val_idx, test_idx


def save_checkpoint(model, hyperparameters, filename):
    '''
    Save checkpoint
    Args:
        model:
        hyperparameters:
        filename:

    Returns:

    '''
    checkpoint = {
        'hyperparameters': hyperparameters,
        'model_state_dict': model.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model):
    '''
    Load checkpoint
    Args:
        filename:
        model:

    Returns: model, hyperparameters

    '''
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    hyperparameters = checkpoint['hyperparameters']
    return model, hyperparameters

def get_InferNet_save_path(flag, seed, net_root_path, noise_pct , MODEL_NAME):
    model_params_MLP = ""
    if flag:
        if MODEL_NAME == "InferNet_GT2":
            model_params_MLP = net_root_path + "NN_params_infer_synthetic_v2_" + "noise=" + str(noise_pct) + "_seed=" + str(
                seed) + ".pth"
        elif MODEL_NAME == "InferNet_GT3":
            model_params_MLP = net_root_path + "NN_params_infer_GT3_synthetic_v2_" + "noise=" + str(noise_pct) + "_seed=" + str(
                seed) + ".pth"
    else:
        if MODEL_NAME == "InferNet_GT2":
            model_params_MLP = net_root_path + "NN_params_infer_seed=" + str(seed) + ".pth"
        elif MODEL_NAME == "InferNet_GT3":
            model_params_MLP = net_root_path + "NN_params_infer_GT3_seed=" + str(seed) + ".pth"

    return model_params_MLP

def get_MDN_save_path(flag, seed, net_root_path, noise_pct, MODEL_NAME):
    model_params_MLP = ""
    prefix = ""

    if flag:
        prefix = "synthetic_noise="+str(noise_pct) +"_"
    else:
        prefix = "real_"
    model_params_MLP = net_root_path + prefix + MODEL_NAME + "_seed="+ str(seed) + ".pth"

    return model_params_MLP

def get_Input_List(MODEL_NAME):
    '''
    Return INPUT_LIST according to MODEL_NAME
    Args:
        MODEL_NAME:

    Returns:

    '''
    MODEL_LIST = ["GT1(MDN)", "GT2(MDN)", "EMD", "GT1+GT2", "GT1+EMD", "GT2+EMD", "GT1_GT2_EMD"]

    INPUT_LIST = []

    if MODEL_NAME == "GT1(MDN)":
        INPUT_LIST = [1]
    elif MODEL_NAME == "GT2(MDN)":
        INPUT_LIST = [2]
    elif MODEL_NAME == "GT3":
        INPUT_LIST = [3]
    elif MODEL_NAME == "EMD":
        INPUT_LIST = [4]

    elif MODEL_NAME == "GT1+GT2":
        INPUT_LIST = [1, 2]
    elif MODEL_NAME == "GT1+GT3":
        INPUT_LIST = [1, 3]
    elif MODEL_NAME == "GT1+EMD":
        INPUT_LIST = [1, 4]
    elif MODEL_NAME == "GT2+GT3":
        INPUT_LIST = [2, 3]
    elif MODEL_NAME == "GT2+EMD":
        INPUT_LIST = [2, 4]
    elif MODEL_NAME == "GT3+EMD":
        INPUT_LIST = [3, 4]

    elif MODEL_NAME == "GT1_GT2_GT3":
        INPUT_LIST = [1, 2, 3]  # 注意和MODEL_NAME要对应起来！
    elif MODEL_NAME == "GT1_GT2_EMD":
        INPUT_LIST = [1, 2, 4]  # 注意和MODEL_NAME要对应起来！
    elif MODEL_NAME == "GT1_GT3_EMD":
        INPUT_LIST = [1, 3, 4]  # 注意和MODEL_NAME要对应起来！
    elif MODEL_NAME == "GT2_GT3_EMD":
        INPUT_LIST = [2, 3, 4]  # 注意和MODEL_NAME要对应起来！
    else:
        assert "Wrong Model Name!"

    return INPUT_LIST


def get_metric_save_path(ARTIFICIAL, MODEL_NAME):

    if ARTIFICIAL:
        metric_file_name = "metric_saved/" + "synthetic_"+ MODEL_NAME + ".csv"
    else:
        metric_file_name = "metric_saved/" + "real_"+ MODEL_NAME + ".csv"

    return metric_file_name

def save_performance(ARTIFICIAL, seed, MODEL_NAME, performance, metric_list):
    metric_pd = pd.DataFrame()

    metric_pd.loc[seed, metric_list[0]] = performance[0]
    metric_pd.loc[seed, metric_list[1]] = performance[1]
    metric_pd.reset_index(inplace=True)

    metric_output_name = get_metric_save_path(ARTIFICIAL, MODEL_NAME)

    if os.path.exists(metric_output_name):
        existing_metric = pd.read_csv(metric_output_name)
        updated_metric = pd.concat([existing_metric, metric_pd], ignore_index=True, axis=0)
        updated_metric.to_csv(metric_output_name, header=True, index=False)
    else:
        metric_pd.to_csv(metric_output_name, header=True, index=False)
