#!/usr/bin/env python
# coding: utf-8
import json



import os
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy
import analyse.image_from_mask
import csv
import time
import pandas as pd
import pathlib
import rasterio



#creates a csv file with error rates and other measures for all images



def create_csv_with_prediction_info(experiment_settings_dict):
    """
    @ side effect: creates statistics_model_MODELNAME_dataset_DATASETNAME.csv, med header ["filename","error_rate"]+["iou_"+category for category in categories]+["confusionmatrix"]

    #output från denna function kan sedan gås igenom før at skapa
    #confusionmatrix_model_MODELNAME_dataset_DATASETNAME_subset_SUBSET
    #errorrate_and_iou_model_MODELNAME_dataset_DATASETNAME_subset_SUBSET
    """
    # create list of categories e.g ["Baggrund","Bygning","Skov","Vej","Vand","Mark"]
    categories = [line.rstrip() for line in open(experiment_settings_dict["path_to_codes"]).readlines()]

    path_to_labels= experiment_settings_dict["path_to_labels"]
    test_set= experiment_settings_dict["dataset_folder"].name  #the name of the dataset e.g 'AmagerLangeland1000p160mmRTO' or CAMVID_TESTIMAGES
    
    im_dir = experiment_settings_dict["path_to_images"]  #the path to the folder for the images




    predictions_dir = experiment_settings_dict["prediction_folder"]

    #make sure folder exists
    predictions_dir.mkdir(parents=True, exist_ok=True)
    # We use a dictionary to save both path and info about if a subfolder is used

    selection= experiment_settings_dict["path_to_all_txt"] #TODO! this part is rather cryptic, should be rewritten! so its understandable

    if "ignore_index" in  experiment_settings_dict:
        ignore_index = experiment_settings_dict["ignore_index"]
    else:
        #sdfi default ignore index.
        ignore_index = 255

    if selection ==None:
        all_file_dicts = []
        #1. get paths for all images
        #print(" images in im_dir : "+str(im_dir))
        #need to be able to handle subfolders and absence of subfolders

        for file_or_folder in os.listdir(im_dir):
            path=im_dir/file_or_folder
            if os.path.isdir(path):
                for file in os.listdir(path):
                    file_path = path/ file
                    all_file_dicts.append({"file_path":pathlib.Path(file_path),"in_subfolder":file_or_folder})
            else:
                all_file_dicts.append({"file_path":pathlib.Path(path),"in_subfolder":False} )

        #os.path.splitext(all_files[0])[-1] splits the path in path and file ending (e.g .png or .tif)
        image_file_dicts =[a_file_dict for a_file_dict in all_file_dicts if os.path.splitext(a_file_dict["file_path"])[-1] in [".png",".tif"] ]
    else:
        print("creating csv for the images in:"+str(selection))
        image_file_dicts=[]
        with open(selection) as f:
            selected_files=f.readlines()
        print("selected_files:"+str(selected_files))
        for selected_file in selected_files:
            selected_file=selected_file.rstrip() #remove \n spaces and the like
            #ignore empty lines
            if selected_file:
                # need to be able to handle subfolders and absence of subfolders
                #if there is no block subfolder
                if (pathlib.Path(selected_file).parent.name == ""):
                    folder =False
                else:
                    folder = pathlib.Path(selected_file).parent.name
                #path=im_dir/selected_file
                #image_file_dicts.append({"file_path": pathlib.Path(path).as_posix(), "in_subfolder": folder})
                image_file_dicts.append({"file_path": selected_file, "in_subfolder": folder})
            





    #create csv file writer model_performance_on_each_image_csv
    #store the path in the dictionary so we can acess it when creating the pdf report
    log_folder = experiment_settings_dict["log_file"].parent


    if log_folder.exists():
        print("log_folder exists, if you fail to create a file in this foler it is becaus your filename is to long")

    #THIS WORKS, if longer filenames are risky!
    #experiment_settings_dict["model_performance_on_each_image_csv"] =log_folder/pathlib.Path( "short_file.csv" )#experiment_settings_dict["log_file"].name.replace(".csv","model_on_dataset_"+test_set+".csv"))
    #experiment_settings_dict["model_performance_on_each_image_csv"] =log_folder/pathlib.Path(experiment_settings_dict["log_file"].name.replace(".csv","model_on_dataset_"+test_set+".csv"))

    csv_file_name = experiment_settings_dict["model_performance_on_each_image_csv"]
    print("crating csv_file_name:"+str(csv_file_name))
    delimiter=";"
    with open(csv_file_name, 'w', encoding='UTF8',newline='') as f:
        csv_writer =csv.writer(f, delimiter=delimiter)
        header = ["filename","error_rate"]+["iou_"+category for category in categories]+["confusionmatrix"]
        csv_writer.writerow(header)# write the header


    #iterate over the images, and write one row to the csv for each image
    file_nr=0
    files_with_missing_labels_or_predictions=0
    for image_file_dict in image_file_dicts:
        

        image_file_path=image_file_dict["file_path"]
        #overwrite info to same row everytime (no carage return)
        print("working on file nr : "+str(file_nr) + " out of : "+str(len(image_file_dicts)) +" Filename : "+str(image_file_path) +   '\r', end="")


        #1. LOAD THE PATHS TO PREDICTION IMAGE and LABEL IMAGE
        #when finding the path to the predictions and labels, we need to check if image was in a a block folder or not as these block folders also will exist in the label folder and prediction folder
        if image_file_dict["in_subfolder"]:
            subfolder =image_file_dict["in_subfolder"]
            file_name = subfolder +"/"+ os.path.split(image_file_path)[-1]
        else:
            file_name =  os.path.split(image_file_path)[-1]



        prediction_mask_path = predictions_dir /file_name
        label_mask_path = path_to_labels/(pathlib.Path(file_name).stem + experiment_settings_dict["label_image_type"])

        if os.path.exists(prediction_mask_path) and  os.path.exists(label_mask_path):
            #load predictions and labels
            if "prediciton_image_is_probs" in experiment_settings_dict and experiment_settings_dict["prediciton_image_is_probs"]:
                #prediction image is showing the raw probabilities. in order to get teh predictions we need to do argmax
                pred_mask_img = rasterio.open(prediction_mask_path).read()
                numpy_prediction_mask = numpy.array(pred_mask_img,dtype=float).argmax(axis=0).astype(numpy.uint8)
            else:
                pred_mask_img = Image.open(prediction_mask_path)
                numpy_prediction_mask = numpy.array(pred_mask_img)
            label_im = Image.open(label_mask_path)
            numpy_label_mask = numpy.array(label_im)

            ignorer_labels_mask = numpy_label_mask == ignore_index

            # set the area with ignorer index to the ignorer index value as in the label mask
            numpy_prediction_mask[ignorer_labels_mask] = ignore_index
            # pred_mask_img = pred_mask_img_with_ignore_index[ignorer_labels]

            nr_of_ignorer_pixels = sum((ignorer_labels_mask * 1).flatten())


            #print("nr of ignorer index pixels = {}".format(nr_of_ignorer_pixels))


            ##
            """
            import matplotlib.pyplot as plt
            plt.imshow(numpy_prediction_mask, cmap="tab20")
            plt.show()

            # plt.imshow(ignorer_labels, cmap="tab20")
            plt.imshow(numpy_label_mask, cmap="tab20")
            plt.show()
            """


            #IOU SCORES for each category
            iou_scores=[0]*len(categories)

            for category in range(len(categories)):
                label_mask = numpy_label_mask ==category
                prediction_mask =numpy_prediction_mask==category

                #CREATE UNIONS and INTERSECTIONS OF PREDICTION AND MASK
                intersection_of_prediction_and_label= prediction_mask*label_mask
                union_of_prediction_and_label = numpy.logical_or(prediction_mask, label_mask)

                #CALCULATE IOU
                # remove ignored pixels by setting them to zero
                number_of_true_positive_pixels = numpy.count_nonzero(intersection_of_prediction_and_label)
                # remove ignored pixels by setting them to zero
                category_iou_score = number_of_true_positive_pixels/(numpy.count_nonzero(union_of_prediction_and_label)+0.0000000001) #avoid divide by zero
                iou_scores[category]=category_iou_score
            # print(iou_scores)

            #CONFUSION MATRIX and ERROR RATE
            confusion_matrix = numpy.zeros([len(categories),len(categories)])
            errors=0
            for category_label in range(len(categories)):
                label_mask = numpy_label_mask == category_label
                for category_prediction in range(len(categories)):
                    prediction_mask = numpy_prediction_mask == category_prediction
                    #how many pixels of category_label are classified as category_prediction
                    intersection_of_prediction_and_label = prediction_mask * label_mask
                    nr_of_category_label_pixels_classified_as_category_prediction = numpy.count_nonzero(intersection_of_prediction_and_label)
                    confusion_matrix[category_label,category_prediction]+=nr_of_category_label_pixels_classified_as_category_prediction
                    if category_label !=category_prediction:
                        errors+=nr_of_category_label_pixels_classified_as_category_prediction
            nr_of_pixels=sum(confusion_matrix.flatten())
            # print(confusion_matrix)


            try:
                assert nr_of_pixels ==1000000
            except:
                print("nr_of_pixels:"+str(nr_of_pixels))
                print("should have been 1_000_000")
                print("WARNING all predictions and labels can not be mapped to any of : "+str(categories))
                print("max nr in label is: "+str(numpy_label_mask.max()))
                print("max nr in predictions is: "+str(numpy_prediction_mask.max()))

            error_rate=errors/nr_of_pixels

            #WRITE FILENAME IOU AND ERROR RATES TO csv
            data= [image_file_path,error_rate]+iou_scores +[json.dumps(confusion_matrix.tolist())]
            with open(csv_file_name, 'a', encoding='UTF8', newline='') as f:
                csv_writer = csv.writer(f, delimiter=delimiter)
                csv_writer.writerow(data)  # write the data

            file_nr+=1
        else:
            files_with_missing_labels_or_predictions+=1
            print("prediction or label missing for : "+str(image_file_path))
            print("os.path.exists("+str(prediction_mask_path)+"):"+str(os.path.exists(prediction_mask_path)))
            print("os.path.exists("+str(label_mask_path)+"):" + str(os.path.exists(label_mask_path)))



