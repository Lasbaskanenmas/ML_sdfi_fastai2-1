#!/usr/bin/env python
# coding: utf-8

#example usage:
#

import argparse
import ast
import pathlib
from pathlib import Path
import numpy
import pandas as pd
import copy
from reportlab.platypus import Table

# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("-c", "--Csvfile", help="path/to/statistics.csv",required=True)


def parse_statistics_csv(path_to_statistics_csv):
	print(path_to_statistics_csv)
	path_to_statistics_csv=Path(path_to_statistics_csv)
	df = pd.read_csv(path_to_statistics_csv,sep=";")
	#result={}
	table_content=[]

	#we are only interested in the subsets name, not the complete path
	table_content.append(["subset : " + " ", str(pathlib.Path(df["subset"][0]).name) ])

	for column_name in df.columns:
		#result[column_name] = df[column_name][0]
		if column_name =="subset":
			pass

		elif column_name =="error_rate":
			table_content.append([column_name + " ",str(format(float(df[column_name][0])*100, '.2f'))    ])
		else:
			table_content.append([column_name + " ",str(format(float(df[column_name][0]), '.2f'))    ])

	'''

	table =	Table([["subset :", subset],
			   ["error_rate :",
				str(format(float(statistics_dictionary["error_rate"]) * 100, '.2f')) + " %"],
			   ["iou_building :",
				str(format(float(statistics_dictionary["iou_Bygning"]), '.2f'))],
			   ["iou_background :",
				str(format(float(statistics_dictionary["iou_Baggrund"]), '.2f'))]
			   ])
	'''
	return Table(table_content)



def get_error_rate_from_statistics_csv(path_to_statistics_csv):
	df = pd.read_csv(path_to_statistics_csv,";")
	#result={}


	for column_name in df.columns:
		if column_name =="error_rate":
			return str(format(float(df[column_name][0])*100, '.2f'))
	return "found no error rate"


if __name__ == "__main__":
	args = parser.parse_args()
	csv_file_path = args.Csvfile
	result = parse_statistics_csv(path_to_statistics_csv=csv_file_path)
	print("result:")
	print(result)





