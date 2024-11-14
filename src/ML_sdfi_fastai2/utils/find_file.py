#!/usr/bin/env python
# coding: utf-8


import sys
import os
import shutil
import argparse
import pathlib
from pathlib import Path


from os.path import join

def find_file(search_in,filename_must_contain):
	"""
	@arg search_in: path/to/folder
	@arg filename_must_contain: list of substrings e.g ["something",".jpg"]
	@return: a list of paths to files or folders matching the criteria. e.g located in the correct folder and filename containing the listed substrings
	"""

	onlyfiles = [join(search_in, f) for f in os.listdir(search_in) if os.path.exists(os.path.join(search_in, f))]
	wanted_files= []
	for file in onlyfiles:
		all_substrings_precent_in_file=True
		for substring in filename_must_contain:
			if not substring in file:
				all_substrings_precent_in_file = False
		if all_substrings_precent_in_file:
			wanted_files.append(file)
	return wanted_files


