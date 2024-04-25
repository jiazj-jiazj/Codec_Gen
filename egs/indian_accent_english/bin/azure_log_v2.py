import os
import sys
import tempfile
import json
# import argparse
import azureml
from azureml.core import Workspace, Experiment, Environment
from azureml.core.compute import ComputeTarget
from azureml.core import ScriptRunConfig, RunConfiguration
from azureml.core.runconfig import DockerConfiguration
# from azureml.contrib.core.gjdrunconfig import GlobalJobDispatcherConfiguration
import re
from datetime import datetime
current_time = datetime.now()
timestamp = current_time.strftime("%Y_%m_%d_%H_%M_%S")

# from azureml.widgets import RunDetails
from datetime import datetime
def parser():
    import easydict
    args = easydict.EasyDict({
        "use_gjd": True, #True
        "region": "southcentralus",#"westus2"/"southcentralus"
        "cpu": False,
        "exp_name": "zhijun_pytorch", #PLC_pytorch/PLC_synthesize/PLC_challenge/PLC_multi_task
        "info": "ar libritts tfcodec 16",
        "nb_repro": 1,
        "train_config": "config.yaml",
        "checkpoint_path": None,
        "checkpoint_dir": None,
    })
    return args

if __name__ == '__main__':
    import argparse
    external_parser = argparse.ArgumentParser(description='Process command line parameters.')
    external_parser.add_argument('--info', type=str, default="baseline",
                        help='info of experiments')
    external_parser.add_argument('--config', type=str, default=None,
                        help='config')
    ext_args = external_parser.parse_args()
    print("ues")
    info = ext_args.info
    exp_config = ext_args.config
    args = parser()
    args.info = info
    DATASTORE_DICT = {
        "packet_loss_concealment": 'zhijun_data',
        "Echo_Detection_v2": 'zhijun_data',
    } # datastore need to be registered first on the workspace
    REGION_DICT = {
        "westus2": 'packet_loss_concealment',
        "southcentralus": 'Echo_Detection_v2',
        }
    VC_DICT = {
        "packet_loss_concealment": 'Westus2.PacketLossConcealment2',
        "Echo_Detection_v2": 'Southcentralus.EchoDetection',
        }   
    workspace_name = REGION_DICT[args.region]
    datastore_name = DATASTORE_DICT[workspace_name]
    vc_name = VC_DICT[workspace_name]
    
    # from azureml.core.authentication import InteractiveLoginAuthentication
    # InteractiveLoginAuthentication(force=True)
    # init workspace
    ws = Workspace(
        subscription_id="f1f491ac-0340-4e5d-87b7-47692be1cb31",
        resource_group="IC3_Common_GPU_cluster",
        workspace_name=workspace_name
    )

    # create experiment
    exp = Experiment(workspace=ws, name=args.exp_name)
    output_directory = f'../azure_log/{timestamp}'
    os.makedirs("../azure_log", exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
    
    # Get the last three runs  
    last_runs = list(exp.get_runs())[:15]  
    
    # Download log files for each run  
    for i, run in enumerate(last_runs):   
        details_with_logs = run.get_details_with_logs()  
        exp_dir_value = None
        for log_file in details_with_logs['logFiles']:  
            local_log_file_path = f"{output_directory}/{i}_{log_file.split('/')[-1]}"  
            
            if log_file.endswith('70_driver_log.txt'):  
                run.download_file(log_file, local_log_file_path) 
                local_log_file_path = f"{output_directory}/{i}_{log_file.split('/')[-1]}"  
                # Read the contents of the file  
                with open(local_log_file_path, 'r') as f:  
                    content = f.read()  
                
                # Extract the --exp-dir value using regex  
                exp_dir_pattern = r"'--exp-dir'\s*,?\s*'([^']+)'"  
                exp_dir_match = re.search(exp_dir_pattern, content)  
    
                if exp_dir_match:  
                    exp_dir_value = exp_dir_match.group(1)  
                    exp_dir_value = exp_dir_value.split('/')[-1] 
                    if exp_dir_value =="":
                        exp_dir_value = bef_exp_dir_value.split('/')[-2]                     
                else:  
                    print(f"Run {i + 1} --exp-dir value not found")
                
                os.remove(local_log_file_path)
                break
            
        

        if exp_dir_value == "":
            exp_dir_value = f"run_{i}"
        
        print(f"i:{i}")
        print(exp_dir_value)
        run_output_directory = f"{output_directory}/{exp_dir_value}" 
        # print(f"bef is {run_output_directory}")
        # run_output_directory = run_output_directory[:-30]
        # print(f"aft is {run_output_directory}")
        os.makedirs(run_output_directory, exist_ok=True) 
        for log_file in details_with_logs['logFiles']:    
            local_log_file_path = f"{run_output_directory}/{i}_{log_file.split('/')[-1]}"  
            run.download_file(log_file, local_log_file_path) 
        
        

        # for log_file in details_with_logs['logFiles']:  
        #     print(log_file)
        #     if log_file.endswith('70_driver_log.txt'):  
        #         local_log_file_path = f"{output_directory}/{log_file.split('/')[-1]}"  
        #         run.download_file(log_file, local_log_file_path)  
    
        #         # Read the contents of the file  
        #         with open(local_log_file_path, 'r') as f:  
        #             content = f.read()  
                
        #         # Extract the --exp-dir value using regex  
        #         exp_dir_pattern = r"'--exp-dir'\s*,?\s*'([^']+)'"  
        #         exp_dir_match = re.search(exp_dir_pattern, content)  
    
        #         if exp_dir_match:  
        #             exp_dir_value = exp_dir_match.group(1)  
        #             exp_dir_value = exp_dir_value.split('/')[-1] 
                     
        #         else:  
        #             print(f"Run {i + 1} --exp-dir value not found")
                
                        
        # # metrics = run.get_metrics()  
        # # print(f"Metrics for run {i + 1}:")  
        # # for metric_name, metric_value in metrics.items():  
        # #     print(f"{metric_name}: {metric_value}")  
        # # print("\n")
        
        
# from azureml.core import Run    
# # Get the current run  
# run = Run.get_context()  
  
# # Log a single value metric  
# run.log("accuracy", 0.95)  
  
# # Log a list of values as a metric  
# run.log_list("losses", [0.1, 0.05, 0.02])  
