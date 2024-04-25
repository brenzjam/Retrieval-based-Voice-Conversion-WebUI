import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import pdb
import shutil
from subprocess import Popen
from time import sleep
import threading
from random import shuffle
import traceback
from multiprocessing import cpu_count

from sklearn.cluster import MiniBatchKMeans
import platform
import numpy as np
import faiss

from configs.config import Config

now_dir = os.getcwd()
sr_dict = {'32k':'32000','40k':'40000', '48k':'48000'}
outside_index_root = os.getenv("outside_index_root")
n_cpu = cpu_count()

def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

def get_arg_params(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-se",
        "--save_every_epoch",
        type=int,
        default=5,
        help="checkpoint save frequency (epoch)",
    )
    parser.add_argument(
        "-te", "--total_epoch", type=int, default=20, help="total_epoch"
    )
    parser.add_argument(
        "-pg", "--pretrainG", type=str, default="", help="Pretrained Generator path"
    )
    parser.add_argument(
        "-pd", "--pretrainD", type=str, default="", help="Pretrained Discriminator path"
    )
    parser.add_argument("-g", "--gpus", type=str, default="0", help="split by -")
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=12, help="batch size"
    )
    parser.add_argument(
        "-sr", "--sample_rate", type=str, default='48k', help="sample rate, 32k/40k/48k"
    )
    parser.add_argument(
        "-dp", "--dataset_path", type=str, required=True, help="sample rate, 32k/40k/48k"
    )
    parser.add_argument(
        "-mp", "--models_path", type=str, default="/home/brendanoconnor/Desktop/jammable/Retrieval-based-Voice-Conversion-WebUI/assets/trained_models", help="sample rate, 32k/40k/48k"
    )
    parser.add_argument("-nmp", "--new_model_path", type=str, default="", help="split by -")
    parser.add_argument(
        "-pt", "--process_type", type=str, default="preprocess_audio", help="sample rate, 32k/40k/48k"
    )    
    parser.add_argument(
        "-sw",
        "--save_every_weights",
        type=str,
        default="0",
        help="save the extracted model in weights directory when saving checkpoints",
    )
    parser.add_argument(
        "-v", "--version", type=str, default='v2', help="model version"
    )
    parser.add_argument(
        "-f0",
        "--if_f0",
        type=int,
        default=1,
        help="use f0 as one of the inputs of the model, 1 or 0",
    )
    parser.add_argument(
        "-tfc",
        "--train_from_ckpt",
        type=str,
        default="",
        help="if model data is intended to be written over, 1 or 0",
    )
    parser.add_argument(
        "-np",
        "--num_processes",
        type=int,
        required=True,
        help="Define how many processes are needed for threading",
    )
    parser.add_argument(
        "-l",
        "--if_latest",
        type=int,
        default=0,
        help="if only save the latest G/D pth file, 1 or 0",
    )
    parser.add_argument(
        "-c",
        "--if_cache_data_in_gpu",
        type=int,
        default=1,
        help="if caching the dataset in GPU memory, 1 or 0",
    )
    
    params = parser.parse_args()

    return params

def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True
    

def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

    
    
def save_train_config(trg_model_path, version, sr):
    config = Config()
    
    # this condition was originally written with 'or sr2 == '40k'
    config_path = f"{version}/{sr}.json"
    
    if version == 'v2' and sr == '40k':
        raise NotImplementedError("v2 model with 40k sample rate is not implemented yet")
    
    config_save_path = os.path.join(trg_model_path, "config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(
            config.json_config[config_path],
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
        )
        f.write("\n")
        
        
def train_with_feats(params, preprocessed_files_path, trg_model_path):

    # 生成filelist
    gt_wavs_dir = "%s/0_gt_wavs" % (preprocessed_files_path)
    feature_dir = (
        "%s/3_feature256" % (preprocessed_files_path)
        if params.version == "v1"
        else "%s/3_feature768" % (preprocessed_files_path)
    )
    if params.if_f0:
        f0_dir = "%s/2a_f0" % (preprocessed_files_path)
        f0nsf_dir = "%s/2b-f0nsf" % (preprocessed_files_path)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    spk_id5 = 0
    for name in names:
        if params.if_f0:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if params.version == "v1" else 768
    if params.if_f0:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, params.sample_rate, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, params.sample_rate, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % trg_model_path, "w") as f:
        f.write("\n".join(opt))
    
    save_train_config(trg_model_path,
                      params.version,
                      params.sample_rate)

    cmd = (f"python infer/modules/train/train.py "
                f"-se {params.save_every_epoch} "
                f"-te {params.total_epoch} "
                f"-g {params.gpus} "
                f"-bs {params.batch_size} "
                f"-mp {trg_model_path} "
                f"-dp {params.dataset_path} "
                f"-sr {params.sample_rate} "
                f"-sw {params.save_every_weights} "
                f"-v {params.version} "
                f"-f0 {params.if_f0} "
                f"-l {params.if_latest} "
                f"-c {params.if_cache_data_in_gpu} "
    )
    if params.pretrainG != "":
        cmd += f"-pg {params.pretrainG} -pd {params.pretrainD}"
    print(f'PRINT {cmd}')
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    

def preprocess_dataset(dataset_path, preprocessed_files_path, sr, n_p):
        
    if int(n_p) > 1:
        no_parallel = True
    else:
        no_parallel = False
        
    # get command and execute using multithreading
    cmd = f'python infer/modules/train/preprocess.py {dataset_path} {sr_dict[sr]} {n_p} {preprocessed_files_path} {no_parallel} 3.7'
    print(f'PRINT {cmd}')
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    
    
def extract_f0_feats(preprocessed_files_path):
    cmd = f'python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 {preprocessed_files_path} True'
    print(f'PRINT {cmd}')
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    
    
def extract_uttr_feats(params, preprocessed_files_path):
    cmd = f"python infer/modules/train/extract_feature_print.py cuda:0 1 0 0 {preprocessed_files_path} {params.version} True"
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    

# but4.click(train_index, [exp_dir1], info3)
def train_index(params, preprocessed_files_path, trg_model_path):
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    feature_dir = (
        "%s/3_feature256" % (preprocessed_files_path)
        if params.version == "v1"
        else "%s/3_feature768" % (preprocessed_files_path)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = []

    def process_file(file_path):
        
        first_file_path = os.path.join(feature_dir, listdir_res[0])
        big_npy = np.load(first_file_path)
        return np.concatenate((big_npy, np.load(file_path)), axis=0)

    if False:
        with ThreadPoolExecutor(max_workers=params.num_processes) as executor:
            # Start loading all files in parallel
            futures = [executor.submit(process_file, os.path.join(feature_dir, name)) for name in listdir_res]

            # Load the first file outside the executor to initialize big_npy


            # Iterate over futures as they complete and concatenate
            for future in futures:
                data = future.result()  # This will block until the future is complete
                big_npy = np.concatenate((big_npy, data), axis=0)
    else: 
        for name in sorted(listdir_res):
            phone = np.load("%s/%s" % (feature_dir, name))
            npys.append(phone)
        big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        # yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            # yield "\n".join(infos)

    np.save("%s/total_fea.npy" % trg_model_path, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    # yield "\n".join(infos)
    index = faiss.index_factory(256 if params.version == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    # yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s.index"
        % (trg_model_path, n_ivf, index_ivf.nprobe),
    )
    infos.append("adding")
    # yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        # "%s/added_IVF%s_Flat_nprobe_%s.index"
        # % (trg_model_path, n_ivf, index_ivf.nprobe),
        "%s/probe.index" % trg_model_path,
    )
    infos.append(
        # "成功构建索引 added_IVF%s_Flat_nprobe_%s.index"
        # % (n_ivf, index_ivf.nprobe)
        "%s/probe.index" % trg_model_path,
    )
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(
            "%s/added_IVF%s_Flat_nprobe_%s.index"
            % (trg_model_path, n_ivf, index_ivf.nprobe),
            "%s/%s_IVF%s_Flat_nprobe_%s.index"
            % (
                outside_index_root,
                preprocessed_files_path,
                n_ivf,
                index_ivf.nprobe,
            ),
        )
        infos.append("链接索引到外部-%s" % (outside_index_root))
    except:
        infos.append("链接索引到外部-%s失败" % (outside_index_root))

    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    # yield "\n".join(infos)    


def make_trg_model_path(params):
    
    if params.if_f0:
        f0_type = 'withF0'
    else:
        f0_type = 'noF0'
        
    if params.pretrainG == "":
        pretrain_str = params.pretrainG
    elif "assests/pretrained" in params.pretrainG:
        pretrain_str = "rvcPretrained"
    else:
        pretrain_str = 'pretrained'
        
    model_name = (
        f"{os.path.basename(params.dataset_path)}"
        f"_{pretrain_str}"
        f"_{f0_type}"
        f"_{params.sample_rate}"
        f"_{params.version}"
    )
    trg_model_path = os.path.join(params.models_path, model_name)
    return trg_model_path

    
def main():
    
    params = get_arg_params()
    preprocessed_files_path = os.path.join(params.dataset_path, "preprocessed_files")
    if not os.path.exists(preprocessed_files_path):
        os.makedirs(preprocessed_files_path)
    
    if params.new_model_path != "":
        params.models_path = params.new_model_path

    trg_model_path = make_trg_model_path(params)
    
    
    # Check if model path already exists with no intention of using its checkpoint
    # if os.path.exists(trg_model_path) and params.pretrainG == "":
    #     raise FileExistsError(f"Model path {trg_model_path} already exists, and user has not specified train from a checkpoint \'-tfc 1\'")
    os.makedirs(trg_model_path, exist_ok=True)
    
    if params.process_type == 'preprocess_audio':
        preprocess_dataset(params.dataset_path, preprocessed_files_path, params.sample_rate, params.num_processes)
    elif params.process_type == 'extract_f0_feats':
        extract_f0_feats(preprocessed_files_path)
    elif params.process_type == 'extract_uttr_feats':
        extract_uttr_feats(params, preprocessed_files_path)
    elif params.process_type == 'train_with_feats':
        train_with_feats(params, preprocessed_files_path, trg_model_path)
    elif params.process_type == 'train_index':
        train_index(params, preprocessed_files_path, trg_model_path)
    else:
        raise NotImplementedError("Invalid process type")
    
# python train_from_dataset.py -sr 48k -dp /home/brendanoconnor/Documents/datasets/speech_vctk/m_p226_vctk_2files -np 11
if __name__ == "__main__":
    main()
