#!/usr/bin/env python
# coding: utf-8

#example usage:
#

import argparse
import ast
import pathlib

from pathlib import Path
import numpy
#PDF reportlab imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image,PageBreak, Table

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

from reportlab.lib import colors
#end reportlab imports
page_break= "PageBreak" #"-------------------------------------------------------------"
#reportlab table indexing is x,y

def reportlab_table_from_confusion_matrix(confusion_matrix,categories):
	"""
	@ arg confusion_matrix:  2 dimensional arrray
	@ arg categories e.g ["baggrund","bygning","vej"]

	@return : reportlab table with a confusionmatrix that includes the names of the categories for each row and column
	"""
	print("confusion_matrix STart")
	print(confusion_matrix)
	print("confusion_matrix end")
	print(categories)
	reportlab_table_list=[]
	reportlab_table_list.append([""]+categories)
	for n_category in range(len(categories)):
		category=categories[n_category]
		reportlab_table_list.append([category]+confusion_matrix[n_category])

	style=[('BOX', (0, 0), (-1, -1), 0.25, colors.blue)]
	style.append(('TEXTCOLOR', (2, 1), (-1, 1), colors.blue))  # first row should be blue
	style.append(('TEXTCOLOR', (1, 2), (1, -1), colors.red)) #first column should be red from 2 column
	style.append(('TEXTCOLOR', (2, 2), (-1, -1), colors.cyan)) #everything except first row should be cyan
	#set all x==y cells to green
	for n_category in range(1,len(categories)): #the baggrund label should not be colored
			style.append(('TEXTCOLOR', (n_category+1, n_category+1), (n_category+1, n_category+1), colors.green))
	#change description fontzise to be smaller ,in order to make everything fit in a pdf page
	style.append(('FONTSIZE', (0, 0), (-1, -1), 5))




	return Table(reportlab_table_list,style=style,hAlign='LEFT')





def get_ilustrative_confusion_matrix(path_to_codes):
	"""
	A confusion matrix that shows what the different numbers mean
	"""
	categories = [line.rstrip() for line in open(path_to_codes).readlines()]
	illustrative_matrix=[]
	for category_label in categories:
		illustrative_matrix.append([])
		for category_prediction in categories:
			illustrative_matrix[-1].append("%"+category_label+" predicted as "+category_prediction)
	confusion_matrix_table = reportlab_table_from_confusion_matrix(illustrative_matrix,categories)

	return Table([["Guide til fortolkning af confusion matrix"],
				  [confusion_matrix_table]], style=[
		('BOX', (0, 0), (-1, -1), 0.25, colors.blue)


	])
	"""

	illustrative_matrix = [[""]+categories]
	for category_label in categories:
		row = [category_label]
		illustrative_matrix.append(row)
		for category_prediction in categories:
			illustrative_matrix[-1].append("% av "+category_label+"pixlar der blivit klassificerade som "+category_prediction)
	return Table([["Guide til fortolkning af confusion matrix"],
				  [Table([["","bygning","baggrund"],
		   ["bygning","% af byggningpixels klassificered som bygning","% af byggningpixels klassificered som baggrund"],
		   ["baggrund","% af baggrundpixels klassificered som bygning","% af baggrundpixels klassificered som baggrund"]],style=[
			# indexing is 0 based x,y
			# green  building-> buliding
			# red background->building
			# blue  building-> background
			# black  background->background
			##('TEXTCOLOR', (1, 1), (1, 1), colors.green),
			##('TEXTCOLOR', (2, 1), (2, 1), colors.red),
			##('TEXTCOLOR', (1, 2), (1, 2), colors.blue),
			('BOX', (0, 0), (-1, -1), 0.25, colors.blue)],hAlign='LEFT')]],style=[
				('BOX', (0, 0), (-1, -1), 0.25, colors.blue)

			])
	
	return Table([["Guide til fortolkning af confusion matrix"],
				  [Table([["","bygning","baggrund"],
		   ["bygning","% af bygningpixels klassificeret som bygning","% af bygningpixels klassificeret som baggrund"],
		   ["baggrund","% af baggrundpixels klassificeret som bygning","% af baggrundpixels klassificeret som baggrund"]],style=[
			# indexing is 0 based x,y
			# green  building-> buliding
			# red background->building
			# blue  building-> background
			# black  background->background
			('TEXTCOLOR', (1, 1), (1, 1), colors.green),
			('TEXTCOLOR', (2, 1), (2, 1), colors.red),
			('TEXTCOLOR', (1, 2), (1, 2), colors.blue),
			('BOX', (0, 0), (-1, -1), 0.25, colors.blue)],hAlign='LEFT')]],style=[
				('BOX', (0, 0), (-1, -1), 0.25, colors.blue)

			])
	"""


def save_pdf(text_or_images,pdf_filename_postfix,model,dataset,benchmark_dataset,experiment_settings_dict,convert_to_jpg=True):
	"""
	given a list of text and  images, create a pdf that visualize the info
	the string "PageBreak" is treated as a special case meaning pdfvisualization should continue on new page
	"""


	#setting up pdflab
	print("generating pdf..")
	reportname = pdf_filename_postfix+"_"+model+".pdf"
	styles = getSampleStyleSheet()
	styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
	doc = SimpleDocTemplate(reportname, pagesize=letter,
							rightMargin=72, leftMargin=72,
							topMargin=72, bottomMargin=18)

	#fill pdf with content
	Story = []

	if "show_intro" in experiment_settings_dict and experiment_settings_dict["show_intro"]:
		Story.append(Paragraph('<b>Performance of ML model on Dataset</b>', styles["Normal"]))
		Story.append(Spacer(1, 12))
		Story.append(Spacer(1, 12))
		Story.append(Paragraph('<i>A machine learning model is created by training a certain <b>model architecture</b> (nr of neurons in different layers etc) on a certain <b>dataset</b> (collection of images and labels).</i>', styles["Normal"]))
		Story.append(Spacer(1, 12))
		Story.append(Paragraph('<i>The <b>model name</b> corresponds to the trained model.</i>', styles["Normal"]))
		Story.append(Spacer(1, 12))
		Story.append(Paragraph("In order the <b>evaluate the performance</b> of the model it needs to be tested on a separate dataset that includes images <b>not seen during training.</b>", styles["Normal"]))
		Story.append(Spacer(1, 12))
		Story.append(Spacer(1, 12))
		Story.append(Table([['Model: ',str(model)],
		 ['Trained on dataset:',str(dataset)],
		 ['Validated on benchmarkdataset:',pathlib.Path(benchmark_dataset).name],
		 ['Images visualized in format:',"JPEG"]],hAlign='LEFT'))
		Story.append(Spacer(1, 12))
		Story.append(Spacer(1, 12))
		Story.append(Paragraph('<b>Performance</b>', styles["Normal"]))
		Story.append(Paragraph('A dataset consists of a set of images, where all pixels have a certain label-class (building, road, background etc)', styles["Normal"]))
		Story.append(Spacer(1, 12))
		Story.append(Paragraph('<i>The main measure on how well a model performs on a dataset consists of the % of the total number of pixels in the dataset it get wrong.</i>', styles["Normal"]))
		Story.append(Spacer(1, 12))
		Story.append(Paragraph('<i>This % number is called the error rate.</i>', styles["Normal"]))
		Story.append(Spacer(1, 12))






	# images and text
	for i in range(len(text_or_images)):
		
		if type(text_or_images[i]) ==type(str()):
			if text_or_images[i] =="PageBreak":
				Story.append(PageBreak())
			elif text_or_images[i] =="Spacer":
				Story.append(Spacer(1, 12))
			else:
				Story.append(Paragraph(text_or_images[i], styles["Normal"]))
		else:
			print(type(text_or_images[i]).__name__)
			if type(text_or_images[i]).__name__ in ["PngImageFile","JpegImageFile"] or type(text_or_images[i]).__name__ == "Image":
				print("parse as image")
				pilimage = text_or_images[i]


				pilimage = pilimage.convert("RGB")
				img2 = BytesIO()
				pilimage.save(img2, 'JPEG')
				img2.seek(0)

				reportlabimage =Image(img2)
				#make sure the image keep its aspect ratio but dont become to wide for the pdf
				reportlabimage._restrictSize(6.0 * inch, 4 * inch)
				Story.append(reportlabimage)
				#Story.append(Image(img2, 4 * inch,
				#				   2 * inch))  # image should be   as wide as height since there are two images side by side
				#Story.append(Image(img2, 4 * inch,
				#				   2 * inch))  # image should be   as wide as height since there are two images side by side
			elif type(text_or_images[i]).__name__ == "Table":
				print("parse as Table")
				Story.append(text_or_images[i])
			else:
				print("assuming reportlab format allready "+str(text_or_images[i]))
				Story.append(text_or_images[i])


	doc.build(Story)
	print("save .pdf as :"+reportname)