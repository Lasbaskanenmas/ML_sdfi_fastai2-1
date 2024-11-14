import sdfi_dataset
from pathlib import Path
import train
import utils.utils as sdfi_utils
import argparse
import pathlib
from fastcore.xtras import Path
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import rasterio
import time
import torch
import os
import yaml
def save_as_image(path,data,new_meta):
    """
    :param path: path to the file to create
    :param data: preeictions or probabilities that we want to save
    :param meta_data: meta data that describves the new images geo-ref
    :return: None
    """
    if len(data.shape)==2:
        #when saving predictions we want to save a 2 dimensional array
        #ad an extra dimension of there only are 2
        #rasterios write operation demands a 3dim array
        data= np.expand_dims(data,axis=0)
        #there will only be one value for each 2 dimensional position
        new_meta["count"]=1
        #the classification is an unsigned integer
        new_meta["dtype"]=np.uint8
    else:
        #when saving the probability for each class we save a 3 dimensional array
        #there will be many values for a certain 2 dim position
        new_meta["count"]=data.shape[0]

        #probabilities are floats
        new_meta["dtype"]=np.float32

    with rasterio.open(path, "w", **new_meta) as dest:
            dest.write(data)


def merge(file_name,input_folders,weights,output_folder,save_probs,save_pred):
    first = True
    print("merging :"+str(input_folders))
    print("to: "+str(output_folder)+str(file_name))


    for i in range(len(input_folders)):
        input_folder = input_folders[i]
        weight= weights[i]
        with rasterio.open(input_folder/Path(file_name)) as src:
            # make a copy of the geotiff metadata so we can save the prediction/probabilities as the same kind of geotif as the input image
            new_meta = src.meta.copy()
            new_xform = src.transform
            numpy_data= src.read()

        tmp_result = numpy_data
        if first:
            result = tmp_result*weight
            first = False
        else:
            result = result + tmp_result*weight
    if save_pred:
        print("saving pred")
        pathlib.Path(output_folder+"_pred").mkdir(parents=True, exist_ok=True)
        save_as_image(path=pathlib.Path(output_folder+"_pred")/Path(file_name.replace("PROBS_","")) ,data=np.array(result.argmax(axis=0),dtype=np.uint8),new_meta=new_meta)
    if save_probs:
        pathlib.Path(output_folder+"_probs").mkdir(parents=True, exist_ok=True)
        print("saving probs")
        save_as_image(path=pathlib.Path(output_folder+"_probs")/Path(file_name),data=result,new_meta=new_meta)

def get_files(folder,suffix= ".tif"):
    paths_to_files = [pathlib.Path(os.path.join(folder, f)) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    paths_to_files = [path for path in paths_to_files if path.suffix == suffix]
    return paths_to_files

def main(config):
    with open(config, 'r') as f:
        experiment_settings_dict = yaml.safe_load(f)


    output_folder= experiment_settings_dict["output_folder"]
    input_folders = [experiment_settings_dict["input_folder1"],experiment_settings_dict["input_folder2"]]
    weights = experiment_settings_dict["weights"]
    files = get_files(input_folders[0])
    for file in files:
        file_name = file.name
        merge(file_name,input_folders,weights,output_folder,save_probs=experiment_settings_dict["save_probs"],save_pred=experiment_settings_dict["save_pred"])

    print("DONE processing all images in :"+str(input_folders[0]))

if __name__ == "__main__":
    """
    Given the folders were probability-images have been stored from different models , merge these to a single probability image and stores the probabilities and combined predictions.
    The probabilities of the different models can be weighted so that one model influence the output more than another.

    """
    usage_example="example usage: \n "+r"python merge_predictions_from_different_models.py --config ./configs/example_merge.ini"
    # Initialize parser
    parser = argparse.ArgumentParser(
        epilog=usage_example,
        formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument("-c", "--config", help="one or more paths to experiment config file",nargs ='+',required=True)
    args = parser.parse_args()

    for config in args.config:
        main(config)




