import re
from subprocess import Popen, PIPE
import threading
import os
import pdb
import time
from time import sleep

sample_rates = ['48k']
versions = ['v2']
num_cpus = 11
num_epochs = [200] #TODO: Mechanism for getting network to continue training
use_f0 = [1]
models_path = "assets/trained_models"
save_latest = 1
use_cache = 0
save_all_weights = 1
batch_size = 12
total_commands = 0
model_num = 0
g_ckpt_paths = ["rvc_pretrained"] # "rvc_pretrained" will train starting from the relevant RVC-pretrained models. "" will train from scratch
show_command_only = False


if g_ckpt_paths[0] != "rvc_pretrained":
    for g_ckpt_path in g_ckpt_paths:
        assert bool(re.search(r'G_.*?\.pth', g_ckpt_path)), "Checkpoint paths must point to the generator of the model, which should be in the format of \'G_.*.pth\'"

def ensure_list(variable):
    if not isinstance(variable, list):
        return [variable]
    return variable


# datasets_dir = '/home/brendanoconnor/Documents/datasets/speech_vctk'
# dataset_paths = [os.path.join(datasets_dir, name) for name in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, name))]

dataset_paths = ['/home/brendanoconnor/Documents/datasets/fameplay_meky/no_silence_no_44kHz_withActThresh_0.9']
# dataset_paths = ['/home/brendanoconnor/Documents/datasets/speech_vctk/m_p226_vctk_1files']
# dataset_paths = ['/home/brendanoconnor/Documents/datasets/speech_vctk/m_p226_vctk_100_files',
#                  '/home/brendanoconnor/Documents/datasets/speech_vctk/f_p228_vctk_100_files',
#                  '/home/brendanoconnor/Documents/datasets/singing_inton/f_562082160_inton_11mins',
#                  '/home/brendanoconnor/Documents/datasets/singing_inton/m_1473212998_inton_11mins'
#                  ]


preprocess_task = [
                    'preprocess_audio',
                    'extract_f0_feats',
                    'extract_uttr_feats',
                    ]

training_task = [
                'train_with_feats',
                #  'train_index', # must fix the index training script first
                 ]


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True
    
    
def do_task(cmd):
    try:
        start_time = time.time()
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        # return_code = p.wait()
        output, errors = p.communicate()
        
        # Decode the output and error from bytes to string if necessary
        output = output.decode('utf-8') # outputs everything the process would have output in the terminal
        errors = errors.decode('utf-8') # outputs any errors that would have been printed in the terminal

        # Print the output and errors
        print("Output:")
        print(output)
        print("Errors:")
        print(errors)

        # Get the return code of the process
        return_code = p.returncode
        print("Return Code:", return_code)
        time_taken = time.time() - start_time
        hours = int(time_taken // 3600)
        minutes = int((time_taken % 3600) // 60)
        seconds = int(time_taken % 60)
        
        print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")
            
        # done = [False]
        # threading.Thread(
        #     target=if_done,
        #     args=(
        #         done,
        #         p,
        #     ),
        # ).start()
    except Exception as e:
        
        print(f"Failed to complete model num {model_num}, at command: {cmd}")
        print('due to error:', e)
    

for dsp in dataset_paths:
    
    for sr in sample_rates:
        
        for if_f0 in use_f0:
            if if_f0:
                f0_str = "f0"
            else:
                f0_str = ""
                
            for process_type in preprocess_task:
                total_commands += 1
                cmd = (f"python custom_process_train.py "
                    f"-sr {sr} "
                    f"-dp {dsp} "
                    f"-mp {models_path} "
                    f"-pt {process_type} "
                    f"-f0 {if_f0} "
                    f"-np {num_cpus} "
                )
                
                print(f"Model num {model_num}, command: {cmd}")
                do_task(cmd)
            
            for total_epochs in num_epochs:
                epoch_save_freq = total_epochs//10
                
                for g_ckpt_path in g_ckpt_paths:
                    
                    for version in versions:
                        if g_ckpt_path == "rvc_pretrained":
                            gp_str = f" -pg assets/pretrained_{version}/{f0_str}G{sr}.pth"
                            dp_str = f" -pd assets/pretrained_{version}/{f0_str}D{sr}.pth"
                            new_model_path = f"assets/pretrained_{version}/{f0_str}D{sr}"
                        elif g_ckpt_path != "":
                            new_model_path = os.path.join(models_path,
                                                        os.path.dirname(g_ckpt_path),
                                                        os.path.splitext(os.path.basename(g_ckpt_path))[0][2:] # go from second character to remove reference to G or D
                                                        
                                                    )
                            os.makedirs(new_model_path, exist_ok=True)
                            gp_str = f" -pg {os.path.join(models_path, g_ckpt_path)} "
                            dp_str = "-pd " +re.sub(r'G_(.*?\.pth)', r'D_\1', os.path.join(models_path, g_ckpt_path))
                        else:
                            gp_str = ""
                            dp_str = ""
                            
                        for process_type in training_task:
                            total_commands += 1
                            
                            model_num += 1
                            cmd = (f"python custom_process_train.py "
                                f"-se {epoch_save_freq} "
                                f"-te {total_epochs} "
                                f"-bs {batch_size} "
                                f"-sr {sr} "
                                f"-dp {dsp} "
                                f"-mp {models_path} "
                                f"-pt {process_type} "
                                f"-sw {save_all_weights} "
                                f"-v {version} "
                                f"-f0 {if_f0} "
                                f"-np {num_cpus} "
                                f"-l {save_latest} "
                                f"-c {use_cache} "
                                f"-nmp {new_model_path} "
                                )
                            cmd += gp_str + dp_str                     
                            print(f"Model num {model_num}, command: {cmd}")
                            if show_command_only:
                                sleep(1)
                                os._exit(0)
                            do_task(cmd)