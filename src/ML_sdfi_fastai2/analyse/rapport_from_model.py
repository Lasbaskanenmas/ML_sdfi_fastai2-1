#!/usr/bin/env python
# coding: utf-8
#import ad_grandparent_directory_to_path #makes it posible to find the sdfe folder
import sys
import argparse



import analyse.make_prediction_csv as make_prediction_csv
import pathlib
import analyse.rapport as rapport



	
def main(experiment_settings_dict):
    """
    :param experiment_settings_dict:
    :return: None
    Creates csv_file with performance of model on dataset    
    creating csv files with prediction statistics for the different subsets (land , forstad ,bykerne)
    Also creates pkl and pdf 's with average images
    """
    print(experiment_settings_dict)
    #ad an entry in the dictionary for what csv file name we should store/read error rates iou etc for each image
    try:
        test_set= experiment_settings_dict["dataset_folder"].parent.name  #the name of the dataset e.g 'AmagerLangeland1000p160mmRTO' or CAMVID_TESTIMAGES
    except:
        test_set= experiment_settings_dict["path_to_dataset"].parent.name  #the name of the dataset e.g 'AmagerLangeland1000p160mmRTO' or CAMVID_TESTIMAGES




    #This csv path became to long when generating it automatically formtrasiningexperiment name and dataset name
    #Easy solution is to load the path from ino file instead
    #experiment_settings_dict["model_performance_on_each_image_csv"] =str(experiment_settings_dict["log_file"]).replace(".csv","model_on_dataset_"+test_set+".csv")
	
    if experiment_settings_dict["make_csv"]:
        print("creating csv file with info about how predictions compare to labels")
        # produce csv fil
        make_prediction_csv.create_csv_with_prediction_info(experiment_settings_dict)
        print("###################################")
        print("creating csv files with prediction statistics for the different subsets (land , forstad ,bykerne) ")
        print("creating pdf files with visualisations of the predictions on the dataset : " + str(experiment_settings_dict["dataset_folder"]))
    if experiment_settings_dict["create_subset_files"]:
        rapport.main(experiment_settings_dict)


# rapport.main(modelfile, test_set, subsets, test_set_dir, predictions_dir, test_lb_dir)


# make raport
# make raport


if __name__ == "__main__":
	args = parser.parse_args()
	if str(args.MakeCsv) == "True":
		make_prediction_information_csv = True
	else:
		try:
			assert(str(args.MakeCsv) == "False")
		except:
			sys.exit("--MakeCsv should be set to 'True or 'False' ")
		make_prediction_information_csv = False


	if str(args.Infer) == "True":
		infer = True
	else:
		try:
			assert(str(args.Infer) == "False")
		except:
			sys.exit("--Infer should be set to 'True or 'False' ")
		infer = False


	if args.Dataset:
		dataset =args.Dataset
		print("using dataset : "+str(args.Dataset))
	else:
		dataset = '/mnt/trainingdata/AmagerLangeland1000p160mmRTO/'
	if args.Subset:
		subsets = args.Subset
	else:
		subsets = ["land.txt", "bykerne.txt", "forstad.txt"]
	dataset= pathlib.Path(dataset)
	modelfile= args.Modelfile

	main(dataset=dataset, modelfile=modelfile, infer=infer, subsets=subsets, make_prediction_information_csv=make_prediction_information_csv)



