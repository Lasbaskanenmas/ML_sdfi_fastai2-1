	#!/usr/bin/env python
# coding: utf-8

"""

"""

import pandas as pd

import analyse.reportlab_functions as reportlab_functions
import analyse.image_from_mask as image_from_mask
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
from utils import create_train_txt
from pathlib import Path
from os import listdir
from os.path import join
import os
import pickle
from PIL import Image as PILImage
from reportlab.lib.colors import red
from reportlab.lib.colors import blue
from reportlab.lib.colors import green
import pathlib
import analyse.SDFI_dataset.landsdakkende_testset as landsdakkende_testset
import analyse.rapport_from_model as rapport_from_model
import utils.find_file as find_file



import analyse.parse_statistics_csv as parse_statistics_csv

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
page_break= "PageBreak" #"-------------------------------------------------------------"
new_line= "Spacer"

def find_and_load_pickle_file(search_in,filename_must_contain):
	"""
	Find a uniqe file based on folder location and substrings of its name
	Exits if it fionds more or less than 1 file that match the criteria
	"""
	wanted_files = find_file.find_file(search_in,filename_must_contain)
	if len(wanted_files) != 1:
		print("find_file.py failed for arguments : "+str([search_in,filename_must_contain]))
		print(" not excactly one element in list : "+str(wanted_files) )
		assert(len(wanted_files) == 1)
	else:
		print("unpickling: "+str(wanted_files[0]))
		with open(wanted_files[0], 'rb') as picklefile:
			loaded=pickle.load(picklefile)
		return loaded



def get_training_plot_images(path_to_csv_file):
	"""
	@return a list of PIL images with trainingplots (losss and iou)
	"""
	path_to_csv_file=pathlib.Path(path_to_csv_file)
	log_folder = path_to_csv_file.parents[0]
	paths_to_training_plot=find_file(search_in=log_folder, filename_must_contain=[".png","plot"])

	return [PILImage.open(path_to_training_plot) for path_to_training_plot in paths_to_training_plot]


def get_error_rate(report_cache_directory):
	"""
	:param report_cache_directory:  e.g  e.g path/to/cashe
	:return: the error rate for a model on this subset
	"""
	csv_statistics_file_path = report_cache_directory / pathlib.Path("statistics.csv")
	error_rate = parse_statistics_csv.get_error_rate_from_statistics_csv(csv_statistics_file_path)
	return error_rate



def get_statistics_table(report_cache_directory,subset):
	"""
	:param report_cache_directory: e.g path/to/cashe
	:return: statistics_table: a reportlab table containing things like error rate and iou values
	"""
	subset = pathlib.Path(subset).stem

	csv_statistics_file_path = report_cache_directory / pathlib.Path(subset+"_statistics.csv")
	statistics_table = parse_statistics_csv.parse_statistics_csv(csv_statistics_file_path)
	return statistics_table

def get_confusion_matrix_table(report_cache_directory,subset):
	"""
	:param report_cache_directory: e.g path/to/cashe
	:return: confusion_matrix_table: a reportlab table containing a confusion_matrix
	"""
	subset = pathlib.Path(subset).stem

	csv_confusion_matrix_file_path = report_cache_directory / pathlib.Path(subset+"_confusionmatrix.csv")


	confusion_matrix_dataframe = pd.pandas.read_csv(csv_confusion_matrix_file_path,
													index_col=0)  # index_col=0 parses the first item in each row as the row_index

	confusion_matrix_table = reportlab_functions.reportlab_table_from_confusion_matrix(
		confusion_matrix=confusion_matrix_dataframe.values.tolist(),
		categories=confusion_matrix_dataframe.columns.tolist())

	return confusion_matrix_table


"""
DEPRICATED
def add_confusion_matrix_table_to_report(subset,path_to_subset_txt_file,report_list):
	print("subset: " + subset)
	report_list.append(new_line)
	report_list.append(new_line)
	statistics_table = get_statistics_table(path_to_subset_txt_file)
	confusion_matrix_table = get_confusion_matrix_table(path_to_subset_txt_file)

	report_list.append(
		Table([[statistics_table], [confusion_matrix_table]], style=[('BOX', (0, 0), (-1, -1), 0.25, colors.blue)],
			  hAlign='LEFT'))
"""

def add_fixed_images_to_report(report_list,prediction_folder,labels_folder,images_folder,class_remappings,dataset_folder):
	################################################################
	#### Visualiser fasta billeder                                #
	################################################################


	if (dataset_folder.name =="validationset20230831") or (dataset_folder.name =="all_training_data"):
		import analyse.SDFI_dataset.befaestesle_dataset as befaestesle_dataset
		fixed_images_dict = befaestesle_dataset.get_fixed_set(prediction_base_path=prediction_folder,
																labels_base_path=labels_folder,
																images_base_path=images_folder)
	else:
		fixed_images_dict = landsdakkende_testset.get_fixed_set(prediction_base_path=prediction_folder,
																labels_base_path=labels_folder,
																images_base_path=images_folder)
		#A dictionary of type {"images":{"sommerhuse":['path/to/image.png'],"villahuse":['path/to/image.png'],"industri":['path/to/image.png']},"labels":{"sommerhuse":['path/to/image.png'],"villahuse":['path/to/image.png'],"industri":['path/to/image.png']},"predictions":{"sommerhuse":['path/to/image.png'],"villahuse":['path/to/image.png'],"industri":['path/to/image.png']}}




	#report_list.append(page_break)
	report_list.append(new_line)
	report_list.append("<b>Fixed set of images </b>" )
	#input("SKIPPING FIXE DIMAGES!")
	#return 0

	for subset in fixed_images_dict["images"].keys():

		report_list.append(new_line)
		report_list.append(subset)
		report_list.append(new_line)
		# locate all fixed images
		print("fixed_on_benchmark:"+str(fixed_images_dict))
		for i in range(len(fixed_images_dict["images"][subset])):
			#visualization_image = image_from_mask.masked_image_from_image_prediction_label(
			#	image_path=fixed_images_dict["images"][subset][i], label_path=fixed_images_dict["labels"][subset][i],
			#	prediction_path=fixed_images_dict["predictions"][subset][i],class_remappings=class_remappings)
			visualization_image=image_from_mask.visualize_image_prediction_and_label(image_path=fixed_images_dict["images"][subset][i],label_path=fixed_images_dict["labels"][subset][i],prediction_path=fixed_images_dict["predictions"][subset][i],show=False,save=False,visualization_name="NA",class_remappings=class_remappings)

			report_list.append(visualization_image)
			
		report_list.append(page_break)



def add_average_images_to_report(dataset_used_for_benchmarking,report_list,subset_paths,report_cache,class_remappings,experiment_settings_dict):
	"""
	:param dataset_used_for_benchmarking:
	:param report_list:
	:param subset_paths:
	:param report_cache:
	:param class_remappings: e.g {4:0,7:0} to cahnge al lables/predictions of class 4 and 7 to 0
	:return:
	"""
	################################################################
	####3 Visualiser billeder "i midten" (gennemsnitligt resultat) #
	################################################################
	#locating the average/(middle accuracy) image for each subset
	for subset_path in subset_paths:
		print("subset_path:"+str(subset_path))


		report_list.append("Images with average error rate in subset : " +str(subset_path ))
		report_list.append(new_line)

		subsets_dictionary= find_and_load_pickle_file(search_in=report_cache,
													  filename_must_contain=[".pkl", "indexed_from", "_"+str(subset_path.name)+"_",
																			 Path(dataset_used_for_benchmarking).name])
		# locate the images to show in report
		for i in range(experiment_settings_dict["nr_of_images_to_show"]):
			#visualization_image = image_from_mask.masked_image_from_image_prediction_label(image_path=subsets_dictionary["images"][i], label_path=subsets_dictionary["labels"][i],prediction_path=subsets_dictionary["predictions"][i],class_remappings=class_remappings)
			visualization_image=image_from_mask.visualize_image_prediction_and_label(image_path=subsets_dictionary["images"][i],label_path=subsets_dictionary["labels"][i],prediction_path=subsets_dictionary["predictions"][i],show=False,save=False,visualization_name="NA",class_remappings=class_remappings,names=open(experiment_settings_dict["path_to_codes"]).read().split())
			report_list.append(visualization_image)

		report_list.append(page_break)

def add_worst_images_to_report(dataset_used_for_benchmarking,report_list,subset_paths,report_cache,class_remappings,experiment_settings_dict):
	"""
	:param dataset_used_for_benchmarking:
	:param report_list:
	:param subset_paths:
	:param report_cache:
	:param class_remappings: e.g {4:0,7:0} to cahnge al lables/predictions of class 4 and 7 to 0
	:return:
	"""
	################################################################
	#### Visualizing images with highest number of errors#
	################################################################
	#locating the lowest accuracy image for each subset
	for subset_path in subset_paths:
		print("subset_path:"+str(subset_path))


		report_list.append("Images with max error rate in subset : " +str(subset_path ))
		report_list.append(new_line)

		subsets_dictionary= find_and_load_pickle_file(search_in=report_cache,
													  filename_must_contain=[".pkl", "descending", "_"+str(subset_path.name)+"_",
																			 Path(dataset_used_for_benchmarking).name])
		# locate the images to show in report
		for i in range(experiment_settings_dict["nr_of_images_to_show"]):
			#visualization_image = image_from_mask.masked_image_from_image_prediction_label(image_path=subsets_dictionary["images"][i], label_path=subsets_dictionary["labels"][i],prediction_path=subsets_dictionary["predictions"][i],class_remappings=class_remappings)
			visualization_image=image_from_mask.visualize_image_prediction_and_label(image_path=subsets_dictionary["images"][i],label_path=subsets_dictionary["labels"][i],prediction_path=subsets_dictionary["predictions"][i],show=False,save=False,visualization_name="NA",class_remappings=class_remappings,names=open(experiment_settings_dict["path_to_codes"]).read().split())
			report_list.append(visualization_image)

		report_list.append(page_break)



def prepare_report(experiment_settings_dict):
	"""
	return report_list: images and information about the images in a list for later visualisation in pdf format
	"""
	print("Collect info about error rates etc on all images, create pdf files and .pkl files with info about the images in the pdf files")
	rapport_from_model.main(experiment_settings_dict)

	report_list = []
	subsets=  [subset.name for subset in experiment_settings_dict["paths_to_subset_files"]]



	#error rate for the first (there might be more! subset)
	#report_list.append(Paragraph("<b>Error rate :</b> "+str(get_error_rate(experiment_settings_dict["report_cache_directory"])), styles["Normal"]) )


	#report_list.append(PageBreak())
	report_list.append(Spacer(1, 12))
	#Story.append(Paragraph("Model: "+str(model), styles["Normal"]))
	#Story.append(Paragraph( "Trained on dataset: "+str(dataset), styles["Normal"]))
	#Story.append(Paragraph("Validated on benchmardataset Amager-Langeland", styles["Normal"]))


	
	
	#CONFUSION MATRIX
	report_list.append(reportlab_functions.get_ilustrative_confusion_matrix(path_to_codes= experiment_settings_dict["path_to_codes"]))

	################################################################
	####add confusion matrix, error rates and Iou for each subset to pdf report
	################################################################
	for i_subset in range(len(subsets)):
		subset = subsets[i_subset]
		path_to_subset_txt_file = experiment_settings_dict["paths_to_subset_files"][i_subset]

		#create a blue box with statistics about the models performance on the dataset.
		#The content is organised as a list with one item foir each thign to show inside the blue box
		things_to_show_in_box =[]

		#Statistics
		#Some reports are aimed towards an audience for wich things like iou values are an overkill
		if "show_statistics" in experiment_settings_dict and experiment_settings_dict["show_statistics"]:
			statistics_table = get_statistics_table(experiment_settings_dict["report_cache_directory"],subset)
			things_to_show_in_box.append([statistics_table])


		#Confusion matrix
		report_list.append("Confusion matrix")
		#create a blue box with a confusion matrix describing the models performance on the dataset
		confusion_matrix_table = get_confusion_matrix_table(experiment_settings_dict["report_cache_directory"],subset)
		things_to_show_in_box.append([confusion_matrix_table])

		#Statistics and confusion matrix
		#place both statistics and confusion matrix in a blue box ,on separate rows
		report_list.append(
			Table(things_to_show_in_box, style=[('BOX', (0, 0), (-1, -1), 0.25, colors.blue)],
				  hAlign='LEFT'))

	#EXAMPLE IMAGES
	example_image_introduction = """Examples of images with their labels and classifications
	Error rates and confusion matrixes are good for comparing models with each other, but to really get a feeling for how well a model performs on a dataset you need visualisations of how the model interprets images.
	The following visualizations are divided into two parts.
	1.	Fixed images that are selected because they show a famous place or otherwise interesting area. (these iamges will be the same when comparing reports made with different models)
	2.	Images that are representative for how well the model performed because the error rate on those images are close to the median for the dataset. (these images will differ between reports made with different models)
	The image to the left is the original image, on the image to the right pixels are colorized according to how the classification fits with the label.

	Color explanations
	Correct classifications:
	No color: The label says background (no specific label) and the classification says the same.
	Green: The label says that the pixel is of some class other than background and the classification says the same.
	Wrong classification:
	Blue: The label says background (no specific label) and the classification says something else(e.g building).
	Red: The label says that the pixel is of some class other than background but the classifications says background.
	Cyan: The label and classification says two different kinds of classes other than background (e.g building and road).
	"""
	report_list.append(page_break)
	report_list.append('<b>Examples of images with their labels and classifications</b>') # teste at skifte dennne text ud PETER
	report_list.append('<i>Error rates and confusion matrixes are good for comparing models with each other, but to really get a feeling for how well a model performs on a dataset you need visualisations of how the model interprets images.</i>')
	report_list.append(new_line)
	report_list.append('<i>The following visualizations are divided into two parts.</i>')
	report_list.append(new_line)
	report_list.append('<i>1.	Fixed images that are selected because they show a famous place or otherwise interesting area. (These images will be the same when comparing reports made with different models)</i>')
	report_list.append('<i>2.	Images that are representative for how well the model performed because the error rate on those images are close to the median for the dataset. (These images will differ between reports made with different models)</i>')
	report_list.append(new_line)
	report_list.append('<i>The image to the left is the original image, on the image to the right pixels are colorized according to how the classification fits with the label.</i>')
	report_list.append(new_line)
	report_list.append(new_line)
	report_list.append('<b><i>Color explanations</i></b>')
	report_list.append(new_line)
	report_list.append('<i>Correct classifications:</i>')
	report_list.append(new_line)
	report_list.append('<i>No color: The label says background (no specific label) and the classification says the same.</i>')
	report_list.append(new_line)
	report_list.append('<i>Green: The label says that the pixel is of some class other than background and the classification says the same.</i>')
	report_list.append(new_line)
	report_list.append('<i>Wrong classification:</i>')
	report_list.append(new_line)
	report_list.append('<i>Blue: The label says background (no specific label) and the classification says something else(e.g building).</i>')
	report_list.append(new_line)
	report_list.append('<i>Red: The label says that the pixel is of some class other than background but the classifications says background.</i>')
	report_list.append(new_line)
	report_list.append('<i>Cyan: The label and classification says two different kinds of classes other than background (e.g building and road).</i>')
	report_list.append(page_break)

	report_list.append('<b>Fixed Images</b>' ) # teste at skifte dennne text ud PETER
	report_list.append(new_line)
	report_list.append('In order to be able to compare different models performance its nice to see how they do on a fixed set of images.')
	report_list.append(new_line)
	report_list.append('The following images are shown in all reports and are meant to represent a wide variety of areas e.g urban, suburban and countryside.')
	report_list.append(new_line)


	if experiment_settings_dict["show_fixed_images"]:
		add_fixed_images_to_report(report_list,prediction_folder=experiment_settings_dict["prediction_folder"],labels_folder =experiment_settings_dict["path_to_labels"],images_folder=experiment_settings_dict["path_to_images"],class_remappings=experiment_settings_dict["class_remappings"],dataset_folder = experiment_settings_dict["dataset_folder"])


	report_list.append('<b>Images with error rates close to the median error rate</b>' )
	report_list.append(new_line)
	report_list.append('It is informative to get an idea of how well the model can be assumed to perform on a random image. ')
	report_list.append(new_line)
	report_list.append('We here visualize the images for which the number of wrongly classified pixels is closest to the median for the complete dataset. ')
	report_list.append(new_line)

	add_average_images_to_report(experiment_settings_dict["dataset_folder"],report_list,experiment_settings_dict["paths_to_subset_files"],report_cache=experiment_settings_dict["report_cache_directory"],class_remappings=experiment_settings_dict["class_remappings"],experiment_settings_dict=experiment_settings_dict)

	report_list.append('<b>Images with worst error rates</b>')
	report_list.append(new_line)
	report_list.append(
		'It is informative to get an idea of how bad the model can be assumed to perform in worst case. ')
	report_list.append(new_line)
	report_list.append(
		'We here visualize the images for with max nr of wrongly classified pixels. ')
	report_list.append(new_line)

	add_worst_images_to_report(experiment_settings_dict["dataset_folder"], report_list,
								 experiment_settings_dict["paths_to_subset_files"],
								 report_cache=experiment_settings_dict["report_cache_directory"],
								 class_remappings=experiment_settings_dict["class_remappings"],
								 experiment_settings_dict=experiment_settings_dict)

	report_list.append('<b>dictionary used to create report:</b>')
	report_list.append(str(experiment_settings_dict))


	
	#reportlab_functions.save_pdf(text_or_images=report_list, pdf_filename_postfix="foretnings_report",model=Path(args.Modelfile).name, dataset=Path(args.Dataset).name,benchmark_dataset=dataset_used_for_benchmarking)
	return report_list







