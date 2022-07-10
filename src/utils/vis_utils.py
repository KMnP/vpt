import datetime
import os
import glob
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix
# plt.rcParams["axes.grid"] = False

import warnings
warnings.filterwarnings("ignore")
LOG_NAME = "logs.txt"


def remove_trailing(eval_dict):
    min_num = min([len(v) for k, v in eval_dict.items() if "top5" not in k])
    new_dict ={}
    for k, v in eval_dict.items():
        if "top5" not in k:
            new_dict[k] = v[:min_num]
    return new_dict


def get_meta(job_root, job_path, model_type):
    # get lr, wd, feature-type, dataset
    j_data = job_path.split("/run")[0].split(
        job_root + "/" + model_type)[-1].split("/")
    data_name, feat_type, opt_params = j_data[1], j_data[2], j_data[3]
    lr = float(opt_params.split("_")[0].split("lr")[-1])
    wd = float(opt_params.split("_")[1].split("wd")[-1])
    return data_name, feat_type, lr, wd


def update_eval(line, eval_dict, data_name):        
    if "top1" in line and "top" in line.split(": top1:")[-1]:
        metric = "top"     
    else:
        metric = "rocauc"
    top1 = float(line.split(": top1:")[-1].split(metric)[0])
    eval_type = line.split(" Classification results with ")[-1].split(": top1")[0] 
    eval_type = "".join(eval_type.split("_" + data_name))
    eval_dict[eval_type + "_top1"].append(top1)


def get_nmi(job_path):
    with open(job_path) as f:
        lines = f.readlines()
    nmi_dict = defaultdict(list)
    num_jobs = 0
    log_temp = []
    for l in lines:  #, leave=False):
        if "Rank of current process:" in l:
            num_jobs += 1
        if num_jobs == 2:
            break
        if "Clutering nmi" in l:
            n = l.split("Clutering nmi: ")[-1].split(",")[0]
            a_n = l.split("adjusted nmi: ")[-1].split(",")[0]
            v = l.split("v: ")[-1].split(",")[0]
            nmi_dict["nmi"].append(float(n))
            nmi_dict["a_nmi"].append(float(a_n))
            nmi_dict["v_nmi"].append(float(v))
    return nmi_dict


def get_mean_accuracy(job_path, data_name):
    val_data = torch.load(
        job_path.replace("logs.txt", f"val_{data_name}_logits.pth"))
    test_data = torch.load(
        job_path.replace("logs.txt", f"val_{data_name}_logits.pth"))
    v_matrix = confusion_matrix(
        val_data['targets'],
        np.argmax(val_data['joint_logits'], 1)
    )
    t_matrix = confusion_matrix(
        test_data['targets'],
        np.argmax(test_data['joint_logits'], 1)
    )
    return np.mean(v_matrix.diagonal()/v_matrix.sum(axis=1) ) * 100, np.mean(t_matrix.diagonal()/t_matrix.sum(axis=1) ) * 100


def get_training_data(job_path, model_type, job_root):
    data_name, feat_type, lr, wd = get_meta(job_root, job_path, model_type)
    with open(job_path) as f:
        lines = f.readlines()

    # get training loss per epoch, 
    # cls results for both val and test
    train_loss = []
    eval_dict = defaultdict(list)
#     best_epoch = -1
    num_jobs = 0
    total_params = -1
    gradiented_params = -1
    batch_size = None
    for line in lines:  #, leave=False):
        if "{'BATCH_SIZE'" in line and batch_size is None:
            batch_size = int(line.split("'BATCH_SIZE': ")[-1].split(",")[0])
            
        if "Total Parameters: " in line:
            total_params = int(line.split("Total Parameters: ")[-1].split("\t")[0])
            gradiented_params = int(line.split("Gradient Parameters: ")[-1].split("\n")[0])

        if "Rank of current process:" in line:
            num_jobs += 1
        if num_jobs == 2:
            break
        if "average train loss:" in line:
            loss = float(line.split("average train loss: ")[-1])
            train_loss.append(loss)
        if " Classification results with " in line:
            update_eval(line, eval_dict, data_name)

    meta_dict = {
        "data": data_name,
        "feature": feat_type,
        "lr": float(lr) * 256 / int(batch_size),
        "wd": wd,
        "total_params": total_params,
        "tuned_params": gradiented_params,
        "tuned / total (%)": round(gradiented_params / total_params * 100, 4),
        "batch_size": batch_size,
    }
    v_top1, t_top1 = None, None
    return train_loss, eval_dict, meta_dict, (v_top1, t_top1)


def get_time(file):
    with open(file) as f:
        lines = f.readlines()
    start_time = lines[0].split("[")[1].split("]")[0]
    start_time = datetime.datetime.strptime(start_time, '%m/%d %H:%M:%S')

    end_time = lines[-1].split("[")[1].split("]")[0]
    end_time = datetime.datetime.strptime(end_time, '%m/%d %H:%M:%S')

    per_iter = None
    with open(file) as f:
        lines = f.readlines()

    per_batch = []
    per_batch_train = []
    for line in lines[::-1]:
#         print(line)"Test 6/6. loss: 6.097, "
        if ". loss:" in line and "Test" in line:
            per_iter = line.split(" s / batch")[0].split(",")[-1]
            per_batch.append(float(per_iter))
        if ". train loss:" in line:
            per_iter = line.split(" s / batch")[0].split(",")[-1]
            per_batch_train.append(float(per_iter))
            
    return datetime.timedelta(seconds=(end_time-start_time).total_seconds()), np.mean(per_batch), np.mean(per_batch_train)


def get_df(files, model_type, root, is_best=True, is_last=True, max_epoch=300):
    pd_dict = defaultdict(list)
    for job_path in tqdm(files, desc=model_type):
        train_loss, eval_results, meta_dict, (v_top1, t_top1) = get_training_data(job_path, model_type, root)
        batch_size = meta_dict["batch_size"]
        
        if len(eval_results) == 0:
            print(f"job {job_path} not ready")
            continue
        if len(eval_results["val_top1"]) == 0:
            print(f"job {job_path} not ready")
            continue

        if "val_top1" not in eval_results or "test_top1" not in eval_results:
            print(f"inbalanced: {job_path}")
            continue
                
        for k, v in meta_dict.items():
            pd_dict[k].append(v)
        
        metric_b = "val_top1"
        best_epoch = np.argmax(eval_results[metric_b])

        if is_best:
            for name, val in eval_results.items():
                if "top5" in name:
                    continue
                if len(val) == 0:
                    continue
                if not isinstance(val[0], list):
                    try:
                        pd_dict["b-" + name].append(val[best_epoch])
                    except:
                        pd_dict["b-" + name].append(-1)
                        # ongoing training process
                        print(name, best_epoch, val)
        # last epoch
        if is_last:
            if v_top1 is not None:
                pd_dict["l-val_top1"].append(v_top1)
                pd_dict["l-test_top1"].append(t_top1)
                val = eval_results["val_top1"]
            else:
                for name, val in eval_results.items():
                    if "top5" in name:
                        continue
                    if len(val) == 0:
                        continue
                    pd_dict["l-" + name].append(val[-1])

        pd_dict["best_epoch"].append(f"{best_epoch + 1} | {len(val)}")

        pd_dict["file"].append(job_path)
        total_time, _, _ = get_time(job_path)
        pd_dict["total_time"].append(total_time)

    result_df = None
    if len(pd_dict) > 0:
        result_df = pd.DataFrame(pd_dict)
        result_df = result_df.sort_values(['data', "feature", "lr", "wd"])
    return result_df


def delete_ckpts(f):
    # delete saved ckpts for re
    f_dir, _ = os.path.split(f)
    for f_delete in glob.glob(f_dir + "/*.pth"):
        os.remove(f_delete)
        print(f"removed {f_delete}")


def average_df(df, metric_names=["l-val_top1", "l-val_base_top1"], take_average=True):
    # for each data and features and train type, display the averaged results
    data_names = set(list(df["data"]))
    f_names = set(list(df["feature"]))
    t_names = set(list(df["type"]))
    hp_names = [
        c for c in df.columns if c not in ["data", "feature", "type", "file", "best_epoch"] + metric_names]
    data_dict = defaultdict(list)
    for d_name in data_names:
        for f_name in f_names:
            for t_name in t_names:

                result = df[df.data == d_name]
                result = result[result.feature == f_name]
                result = result[result.type == t_name]
                # take average here
                if len(result) == 0:
                    continue
                data_dict["data"].append(d_name)
                data_dict["feature"].append(f_name)
                data_dict["type"].append(t_name)
                data_dict["total_runs"].append(len(result))
        
                for m in metric_names:
                    if take_average:
                        data_dict[m].append("{:.2f}".format(
                            np.mean([r for i, r in enumerate(result[m])]),
                        ))
                        data_dict[f"{m}-std"].append("{:.2f}".format(
                            np.std([r for i, r in enumerate(result[m])])
                        ))
                    else:
                        data_dict[m].append("{:.2f}".format(
                            np.median([r for i, r in enumerate(result[m])]),
                        ))
                for h_name in hp_names:
                    data_dict[h_name].append(result[h_name].iloc[0])

    df = pd.DataFrame(data_dict)
    df = df.sort_values(["data", "feature", "type"])
    return df


def filter_df(df, sorted_cols, max_num):
    # for each data and features, display only top max_num runs
    data_names = set(list(df["data"]))
    f_names = set(list(df["feature"]))
    t_names = set(list(df["type"]))
    df_list = []
    for d_name in data_names:
        for f_name in f_names:
            for t_name in t_names:
                result = df[df.data == d_name]
                result = result[result.feature == f_name]
                result = result[result.type == t_name]
                if len(result) == 0:
                    continue
                cols = [c for c in sorted_cols if c in result.columns]
                result = result.sort_values(cols, ignore_index=True)

                _num = min([max_num, len(result)])
    #             print(result.iloc[-_num:])
                df_list.append(result.iloc[-_num:])
    return pd.concat(df_list)


def display_results(df, sorted_cols=["data", "feature", "type", "l-val_top1"], max_num=1):
    cols = [c for c in df.columns if c not in []]
    df = df[cols]
    if max_num is not None:
        df = filter_df(df, sorted_cols[3:], max_num)
    return df.sort_values(sorted_cols).reset_index(drop=True)
