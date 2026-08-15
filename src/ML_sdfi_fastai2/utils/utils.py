import configparser
import json
from pathlib import Path
from torchvision.models.resnet import  resnet34, resnet50, resnet152, resnet18
from torchvision.models.inception import inception_v3
import ML_sdfi_fastai2.pytorch_models.models as pytorch_models
import sys


def get_model(model_name):
    """
    :param model_name: the model name as a string
    :return: a function that creates a fastai model or the model name string for special cases
    """

    if  model_name == "resnet18":
        return resnet18
    elif model_name == "resnet34":
        return resnet34
    elif model_name == "resnet50":
        return resnet50
    elif model_name == "resnet152":
        return resnet152
    elif model_name == "inception_v3":
        return inception_v3
    elif model_name == "simple_convnet":
        return pytorch_models.create_custom_model
    elif model_name in ["efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", 
                        "efficientnetv2_rw_s.ra2_in1k", "efficientnetv2_rw_m.agc_in1k", 
                        "tf_efficientnetv2_l.in21k", "tf_efficientnetv2_xl.in21k", "resnet50.a1_in1k"]:
        # Using a timm backbone. This will be handled by the wwf.timm_learner
        return model_name
    elif model_name in ["segformer-b0", "segformer-b1", "segformer-b2", "segformer-b3", 
                        "segformer-b4", "segformer-b5"]:
        # SegFormer models will be treated differently
        return model_name
    elif model_name in ["swin-small-upernet","swin-small-upernet", "swin-base-upernet", "swin-large-upernet"]:
        # Swin + UPerNet models
        return model_name
    elif model_name in [
        # Honest names: these load openmmlab ConvNeXt-V1 + UPerNet (no pretrained V2+UPerNet exists).
        "convnext_tiny_upernet",
        "convnext_small_upernet",
        "convnext_base_upernet",
        "convnext_large_upernet",
        # Legacy "convnextv2_*" names kept for back-compat; they resolve to the same V1 checkpoints.
        "convnextv2_tiny_upernet",
        "convnextv2_small_upernet",
        "convnextv2_base_upernet",
        "convnextv2_large_upernet"]:
        return model_name

    else:
        sys.exit("utils.utils.py get_model(model_name) did not recognize model_name: " + str(model_name))


def load_settings_from_config_file(config_file_path):
    """
    :param: config_file_path: e.g "path/to/myexperiment.ini"
    :return: dictionary with all settings needed for training/doing inference, e.g modeltype ,weights to load or dataset to train on
    """

    print("##############################################################################################################################")
    print("######################################### PARSING THE SETTINGS FILE ##########################################################")

    settings_dictionary={}
    parser = configparser.ConfigParser()
    if not Path(config_file_path).is_file():
        sys.exit(config_file_path+ ": is not a file!, did you give the correct path?")
    parser.read(config_file_path)
    sections = parser.sections()
    for section in sections:
        if section =="SUBSETS":
            settings_dictionary["paths_to_subset_files"]=[]
            for key in parser[section]:
                settings_dictionary["paths_to_subset_files"].append(Path(parser[section][key]))
        else:
            print("loading settings in the : "+str(section)+" section")
            for key in parser[section]:
                print("loading value for : "+str(key))
                value_for_key = parser[section][key]

                if key == "model":
                    settings_dictionary[key] = get_model(value_for_key)
                elif key in ["model_to_load","model_used_for_inference"]:
                    if value_for_key == "false":
                        settings_dictionary[key]=json.loads(value_for_key)
                    else:
                        settings_dictionary[key]=Path(value_for_key)
                elif section in ["FOLDERS","DATASET"]:
                    settings_dictionary[key] = Path(value_for_key)
                else:
                    print("key:"+str(key))
                    print("value_for_key:"+str(value_for_key))
                    settings_dictionary[key] = json.loads(str(value_for_key))

    print("######################################### FINNISHED PARSING THE SETTINGS FILE#################################################")
    print("##############################################################################################################################")
    return settings_dictionary

def n_in_from_settings(experiment_settings_dict):
    """
    The total number of input channels the model must accept, checked against the rest of the config.

    n_in is the number of BANDS summed over every datatype, which equals len(means). It is NOT the
    number of datatypes, len(channels). "channels" is a list of per-datatype band-index lists, so
    rgb is [[0,1,2]] (1 datatype, 3 bands) and the all-source config is
    [[0,1,2],[0],[0,1,2],[0],[0],[0]] (6 datatypes, 10 bands).

    Without this check a config whose means/stds do not match its channels still builds a model, and
    the run only dies later on the first forward pass with a conv-weight shape error that says
    nothing about which config key is at fault. Here it fails immediately and names the mismatch.

    :param experiment_settings_dict: the parsed settings dictionary
    :return: n_in, the total number of bands
    """
    means = experiment_settings_dict["means"]
    stds = experiment_settings_dict["stds"]
    channels = experiment_settings_dict["channels"]
    datatypes = experiment_settings_dict["datatypes"]
    bands_described_by_channels = sum(len(bands_of_one_datatype) for bands_of_one_datatype in channels)

    if len(means) != len(stds):
        sys.exit("config error: len(means)=" + str(len(means)) + " != len(stds)=" + str(len(stds)))

    if len(channels) != len(datatypes):
        sys.exit("config error: len(channels)=" + str(len(channels)) + " != len(datatypes)=" +
                 str(len(datatypes)) + ", every datatype needs exactly one list of band indexes")

    if len(means) != bands_described_by_channels:
        sys.exit("config error: len(means)=" + str(len(means)) + " but 'channels'=" + str(channels) +
                 " describes " + str(bands_described_by_channels) + " bands over " + str(len(channels)) +
                 " datatypes. n_in must equal the total number of bands, so means and stds need one"
                 " value per band, not one value per datatype.")

    return len(means)


def save_dictionary_to_disk(experiment_settings_dict):
    """
    Saves the content of the dictionary as a json file
    :param experiment_settings_dict: a dictionary
    :return: None
    """
    json_serializable_dictionary={}
    for key in experiment_settings_dict:
        json_serializable_dictionary[key]= str(experiment_settings_dict[key])

    #Store all job configurations as json file in the log folder
    with open(experiment_settings_dict["log_folder"]/Path(experiment_settings_dict["job_name"]+ "_job_dictionary.json"), "w") as out_file:
        json.dump(json_serializable_dictionary, out_file, indent = 6)


def apply_performance_settings(settings_dictionary):
    """Opt-in, comparability-safe GPU execution speedups (off by default).

    These change only *how fast* kernels run, not the math or precision (bf16 stays bf16),
    so any numerical effect is at floating-point rounding-noise level (~1e-6) -- far below the
    bf16 noise floor and the between-model gaps. Controlled per-run via config keys:
      cudnn_benchmark = true   -> autotune the fastest conv algorithm for the (fixed) input size
      tf32            = true   -> allow TF32 on matmul + cuDNN (helps any fp32-path ops)
    Do NOT combine with make_deterministic() (which sets cudnn.benchmark=False on purpose);
    callers gate this to the non-deterministic path.
    """
    import torch
    if settings_dictionary.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
        print("performance: cudnn.benchmark = True")
    if settings_dictionary.get("tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("performance: TF32 enabled (matmul + cuDNN)")
