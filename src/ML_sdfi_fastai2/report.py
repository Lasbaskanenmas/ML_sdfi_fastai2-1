#!/usr/bin/env python
# coding: utf-8

"""
Creating train reports

python report.py -h to get usage information

TODO: fix folowing bugs: 
1. When several subsets are identified for analyse e.g all.txt and valid.txt , the confusionmatrixes will all be the one belongiogn to the last subset (valid.txt)

"""

#import utils.ad_directories_to_path
import analyse.sdfi_reportlab as sdfi_reportlab
import analyse.reportlab_functions as reportlab_functions

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image,PageBreak, Table
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import sys


import argparse
import utils.utils as sdfi_utils
import utils.create_train_txt
import analyse.plot_training_curves as plot_training_curves
from pathlib import Path
from os import listdir
from os.path import join
import os
import pickle
from PIL import Image as PILImage
import pathlib
import analyse.rapport_from_model as rapport_from_model


# from sdfe.metrics import accuracy # Hacky: torch.serialization expects things related to the Learner to be available in the same place as when the model was exported









def get_training_plot_images(path_to_csv_file,things_to_plot):
	"""
	@return a list of PIL images with trainingplots (losss and iou)
	"""
	plots_to_visualize=[]
	path_to_csv_file=pathlib.Path(path_to_csv_file)
	log_folder = path_to_csv_file.parents[0]

	#LOSS PLOTS ARE HARD TO EVALUATE ON
	path_to_loss_training_plot = str(path_to_csv_file).replace(".csv","loss_plot.jpg")
	values_to_plot=['train_loss', 'valid_loss']
	plot_training_curves.create_plot(log_files=[path_to_csv_file],  output_plot_file=path_to_loss_training_plot,values_to_plot=values_to_plot, image_format="jpg",use_log_format=False,ylim=[0, 3.0],show=False,value_in_name="min")
	plots_to_visualize.append(PILImage.open(path_to_loss_training_plot))

	#things_to_plot = ['valid_accuracy']
	path_to_accuracy_training_plot = str(path_to_csv_file).replace(".csv","accuracy_plot.jpg")
	plot_training_curves.create_plot(log_files=[path_to_csv_file],  output_plot_file=path_to_accuracy_training_plot,values_to_plot=things_to_plot, image_format="jpg",use_log_format=False,ylim=[0, 1.0],show=False,value_in_name="max")
	path_to_accuracy_training_plot_log = str(path_to_csv_file).replace(".csv","accuracy_plot_log.jpg")
	plot_training_curves.create_plot(log_files=[path_to_csv_file],  output_plot_file=path_to_accuracy_training_plot_log,values_to_plot=things_to_plot, image_format="jpg",use_log_format=True,ylim=[0, 1.0],show=False,value_in_name="max")

	plots_to_visualize.append(PILImage.open(path_to_accuracy_training_plot))
	plots_to_visualize.append(PILImage.open(path_to_accuracy_training_plot_log))




	return plots_to_visualize




def add_plots_to_report(report_list,Csvlog,things_to_plot):
	plot_images = get_training_plot_images(Csvlog,things_to_plot)
	
	return plot_images+ report_list

def make_report(experiment_settings_dict):
	
	#create a list with all elements that should be visualized on a PDF
	report_list=sdfi_reportlab.prepare_report(experiment_settings_dict)

	#We can now ad
	#ad trainingplots
	if experiment_settings_dict["show_plots"]:
		report_list= add_plots_to_report(report_list=report_list, Csvlog=experiment_settings_dict["log_file"],things_to_plot = experiment_settings_dict["things_to_plot"])

	#report_list[0].show()
	

	#we only visualize the last two directories in the dataset path
	important_part_of_dataset_path = pathlib.Path(*pathlib.Path(experiment_settings_dict["dataset_folder"]).parts[-2:])


	reportlab_functions.save_pdf(text_or_images=report_list, pdf_filename_postfix=experiment_settings_dict["job_name"],
								 model=Path(experiment_settings_dict["model_used_for_inference"]).name, dataset=experiment_settings_dict["dataset_model_was_trained_on"],experiment_settings_dict=experiment_settings_dict,benchmark_dataset=important_part_of_dataset_path)












if __name__ == "__main__":
    """
    Given one or more dictionaries that defines paths to images labels and infered-segmentation-images, compares the labels and infered images to calculate confusion matrix, error rates, and iou values.
    """
    usage_example="example usage: \n "+r"python report.py --config configs\example_configs\example_report.ini"
    # Initialize parser
    parser = argparse.ArgumentParser(
                                    epilog=usage_example,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument("-c", "--config", help="one or more paths to experiment config file",nargs ='+',required=True)
    args = parser.parse_args()


    for config_file_path in args.config:
        experiment_settings_dict= sdfi_utils.load_settings_from_config_file(config_file_path)
        make_report(experiment_settings_dict)

        
