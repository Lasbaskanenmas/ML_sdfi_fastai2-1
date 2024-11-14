
#Ett skript for at analysere og visualizere hvordan en model presterer på forskelige subsets af et dataset 
#





import os
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy
import csv
import time
import pandas as pd
import analyse.image_from_mask as image_from_mask
import json
import matplotlib.pyplot as plt
import pathlib
import pickle
from pathlib import Path


def get_files_in_subset(subset_file):
    #et dataset kan indeles i forskellige deler der kan analyseras separat : e.g land.txt
    
    with open(subset_file) as a_file:
        files_in_subset = a_file.readlines()
        files_in_subset= [line.strip() for line in files_in_subset] #remove the new lines

    

    return files_in_subset






def get_image_paths(pandas_dataframe,predictions_dir,test_lb_dir,nr_of_images_to_show,property_to_sort_on,ascending,index_of_first_sorted_image_to_show,model_performance_on_each_image_csv_root,experiment_settings_dict):
    """
    @arg pandas_dataframe: a pandas dataframe holding paths to images and inforamtion on how good a classifier is on each image : e.g error rate and confusion matrix
    @arg predictions_dir: path to folder where the predictions were saved
    @arg test_lb_dir: path to folder where the labels are located
    @arg nr_of_images_to_show: how many images that should be located
    @arg property_to_sort_on:  e.g error_rate
    @arg ascending: shuld we sort ascending or descending. e.g lowest values first or highest values first
    @arg index_of_first_sorted_image_to_show: instead of returning the 'nr_of_images_to_show' first images, return the images at index   'index_of_first_sorted_image_to_show' to 'index_of_first_sorted_image_to_show'+'nr_of_images_to_show'
    @model_performance_on_each_image_csv_root: the root directory that the relative paths in the csv file start from
    @experiment_settings_dict: a dictionary desribing report job

    @return: dictionary of type: {"images":[],"labels":[],"predictions":[]}
    holding lists of paths to the images, labels and predictions

    Function sorts the pandas dataframe acording to 'property_to_sort_on' and fills the dictionary with paths to the 'nr_of_images_to_show' first images.
    """

    if ascending:
        sorting_description = "ascending"
    else:
        sorting_description = "descending"
    sorted = pandas_dataframe.sort_values(by=property_to_sort_on, ascending=ascending)

    


    im_paths_list={"images":[],"labels":[],"predictions":[],"buildings":[]}
    #the index_of_first_sorted_image_to_show, makes it posible to show the images in the middle of the sorted list , the 4th quiartile or simular
    for i in range(index_of_first_sorted_image_to_show,index_of_first_sorted_image_to_show+nr_of_images_to_show):
        print("i:"+str(i))
        print("index_of_first_sorted_image_to_show:"+str(index_of_first_sorted_image_to_show))
        print("nr_of_images_to_show:"+str(nr_of_images_to_show))

        print(sorted)
        print(sorted.iloc[i])
        print(sorted.iloc[i][property_to_sort_on])
        print(sorted.iloc[i]["filename"])
        #create a absolute path pointing to the image
        image_file_path= pathlib.Path(model_performance_on_each_image_csv_root) / pathlib.Path( sorted.iloc[i]["filename"])
        #we dont know if the file is in a block folder or not
        #simplest way to find out is to check if the file exists
        file_name= pathlib.Path(image_file_path).name #os.path.split(image_file_path)[-1]
        
        prediction_mask_path = predictions_dir / file_name

        label_mask_path = test_lb_dir/(pathlib.Path(file_name).stem + experiment_settings_dict["label_image_type"])
        building_mask_path = pathlib.Path(experiment_settings_dict["path_to_buildings"])/image_file_path.name
        if not os.path.exists(label_mask_path):
            #file does not exist, we are missing the block folder!
            block_folder = os.path.split(pathlib.Path(image_file_path).parent)[-1]
            label_mask_path = test_lb_dir /block_folder/ file_name
            #print(label_mask_path)
            assert os.path.exists(label_mask_path), str(label_mask_path)+ " does not exist!"
            prediction_mask_path = predictions_dir /block_folder/ file_name
            assert os.path.exists(prediction_mask_path), str(prediction_mask_path) + " does not exist!"

        im_paths_list["images"].append(image_file_path)
        im_paths_list["labels"].append(label_mask_path)
        im_paths_list["predictions"].append(prediction_mask_path)
        im_paths_list["buildings"].append(building_mask_path)
    
    return im_paths_list



def create_pdf_report(dataset,predicted_images_paths_list,nr_of_images_to_show,ascending,property_to_sort_on,index_of_first_sorted_image_to_show,subset_name,report_cache_directory,average_images=False):
    """
    @arg dataset: path to the dataset
    @arg predicted_images_paths_list: a dictionary of type {"images":[],"labels":[],"predictions":[]}
    @arg report_cache_directory: directory where we should save cached files
    @arg average_images: true if we shuld include #indexed_from_" in the saved files name (this means that we only consider the images that the model had average performance on)

    saves a pdf that visualize predictions for the images in the dictionary
    saves a .pkl with info about the images in the pdf
    """

    #the index_of_first_sorted_image_to_show, makes it posible to show the images in the middle of the sorted list , the 4th quiartile or simular

    #make sure that the directory to save the cashed files in exists
    Path(report_cache_directory).mkdir(parents=True, exist_ok=True)



    

    im_list=[]

    if ascending:
        sorting_description = "ascending"
    else:
        sorting_description = "descending"
    print('creating visualized images for the '+str(len(predicted_images_paths_list["images"])) + 'images')
    for i in range(len(predicted_images_paths_list["images"])):

        #pred_mask=numpy.array(Image.open(predicted_images_paths_list["predictions"][i]))==1
        #label_mask= numpy.array(Image.open(predicted_images_paths_list["labels"][i]))==1
        #masks_image = image_from_mask.image_from_masks(label_mask,pred_mask)


        #input_image=Image.open(predicted_images_paths_list["images"][i])
        #composite_image = image_from_mask.image_from_image_and_mask(input_image,numpy.array(masks_image))
        #original_and_masked_images= image_from_mask.create_concatenated_image(input_image,composite_image)

        

        original_and_masked_images= image_from_mask.masked_image_from_image_prediction_label(image_path=predicted_images_paths_list["images"][i], label_path=predicted_images_paths_list["labels"][i], prediction_path=predicted_images_paths_list["predictions"][i],building_path= predicted_images_paths_list["buildings"][i])
        im_list.append(original_and_masked_images)

    dataset_name = Path(dataset).name


    pdf_filename = pathlib.Path(report_cache_directory ) / pathlib.Path(str(nr_of_images_to_show)+"_images_sorted_on_"+sorting_description+"_"+property_to_sort_on+"_"+dataset_name+"_"+subset_name+"_.pdf")
    pkl_filename= pathlib.Path(report_cache_directory ) / pathlib.Path(str(nr_of_images_to_show)+"_images_sorted_on_"+sorting_description+"_"+property_to_sort_on+dataset_name+"_"+"_"+subset_name+"_.pkl")
    
    #all other images are added to the first image and converted to pdf
    #first_im= im_list[0]
    im_list= im_list[1:]
    if average_images:
        pdf_filename = pdf_filename.parent/pdf_filename.name.replace(".pdf","indexed_from_"+str(index_of_first_sorted_image_to_show)+".pdf")
        pkl_filename= pdf_filename.parent/pkl_filename.name.replace(".pkl","indexed_from_"+str(index_of_first_sorted_image_to_show)+".pkl")

    #save pdf and pkl files
    #first_im.convert("RGB").save(pdf_filename, "PDF" ,resolution=100.0, save_all=True, append_images=[im.convert("RGB") for im in im_list])
    pickle.dump(predicted_images_paths_list, open(pkl_filename, "wb" ) )

'''
def DEPRICATED_create_report(pandas_dataframe,predictions_dir,test_lb_dir,nr_of_images_to_show,property_to_sort_on,ascending,index_of_first_sorted_image_to_show,subset=None):





    if ascending:
        sorting_description = "ascending"
    else:
        sorting_description = "descending"
    sorted = pandas_dataframe.sort_values(by=property_to_sort_on, ascending=ascending)
    #create pdf
    im_list=[]
    im_paths_list=[]
    #the index_of_first_sorted_image_to_show, makes it posible to show the images in the middle of the sorted list , the 4th quiartile or simular
    for i in range(index_of_first_sorted_image_to_show,index_of_first_sorted_image_to_show+nr_of_images_to_show):
        #print(sorted.iloc[i])
        #print(sorted.iloc[i][property_to_sort_on])
        #print(sorted.iloc[i]["filename"])
        image_file_path=sorted.iloc[i]["filename"]
        #we dont know if the file is in a block folder or not
        #simplest way to find out is to check if the file exists
        file_name= os.path.split(image_file_path)[-1]
        prediction_mask_path = predictions_dir / file_name
        label_mask_path = test_lb_dir / file_name


        if not os.path.exists(label_mask_path):
            #file does not exist, we are missing the block folder!
            block_folder = os.path.split(pathlib.Path(image_file_path).parent)[-1]
            label_mask_path = test_lb_dir /block_folder/ file_name
            #print(label_mask_path)
            assert os.path.exists(label_mask_path), print(str(label_mask_path)+ " does not exist!" )
            prediction_mask_path = predictions_dir /block_folder/ file_name
            assert os.path.exists(prediction_mask_path), print(str(prediction_mask_path) + " does not exist!")


        pred__mask=numpy.array(Image.open(prediction_mask_path))==1
        label_mask= numpy.array(Image.open(label_mask_path))==1
        masks_image = image_from_mask.image_from_masks(label_mask,pred__mask)
        input_image=Image.open(image_file_path)
        composite_image = image_from_mask.image_from_image_and_mask(input_image,numpy.array(masks_image))
        original_and_masked_images= image_from_mask.create_concatenated_image(input_image,composite_image)
        im_list.append(original_and_masked_images)
    pdf_filename = "./"+str(nr_of_images_to_show)+"_images_sorted_on_"+sorting_description+"_"+property_to_sort_on+"_"+subset+"_.pdf"
    #all other images are added to the first image and converted to pdf
    first_im= im_list[0]
    im_list= im_list[1:]
    if index_of_first_sorted_image_to_show!=0:
        pdf_filename=pdf_filename.replace(".pdf","indexed_from_"+str(index_of_first_sorted_image_to_show)+".pdf")
    first_im.convert("RGB").save(pdf_filename, "PDF" ,resolution=100.0, save_all=True, append_images=[im.convert("RGB") for im in im_list])
'''    
def save_confusion_matrix(average_confusion_matrix,path_to_labels,csv_file_name,experiment_settings_dict):
    # create list of categories e.g ["Baggrund","Bygning","Skov","Vej","Vand","Mark"]
    categories = [line.rstrip() for line in open(experiment_settings_dict["path_to_codes"]).readlines()]
    dataframe=pd.DataFrame(data=average_confusion_matrix, index=categories, columns=categories)
    dataframe.to_csv(csv_file_name,index=True)


    



def save_statistics(statistics_dictionary,csv_file_name):
    try:
        with open(csv_file_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=statistics_dictionary.keys(), delimiter=";")
            writer.writeheader()
            writer.writerow(statistics_dictionary)
    except IOError:
        print("I/O error")


def create_statistics(pandas_dataframe):
    """
    Crate a csv file containing:
    error rate: averaged between images
    confusion matrix: total for all pixles /total number of images
    iou_category1,iou_category2.. iou_categoryn: averaged between images
    """
    number_of_images = pandas_dataframe.shape[0]
    summed_confusion_matrix = None
    summed_error_rate=0
    summed_iou_building = 0
    summed_iou_background = 0
    statistics_dictionary = {}
    

    for data in pandas_dataframe['confusionmatrix']:
        
        if (type(None) == type(summed_confusion_matrix)):
            #first data
            summed_confusion_matrix = numpy.array(json.loads(data),dtype=numpy.int64)
        else:
            summed_confusion_matrix = summed_confusion_matrix+ numpy.array(json.loads(data))
        

    average_confusion_matrix=summed_confusion_matrix/number_of_images

    #to get from raw confusionmatrix to %confusion matrix we must divide by nr of pixels with that label (e.g %correct classified pixels of a ceartain label = pixels_correctly_predicted_as_that_label/pixels_that_actually_have_that_label)
    nr_of_pixels_for_each_label = average_confusion_matrix.sum(axis=1)+0.000000000000001
    print("nr_of_pixels_for_each_label:"+str(nr_of_pixels_for_each_label))
    #divide each row with the number of pixels of that label (divide by to_percentage)
    # multiply with 100 to get %
    #only show 2 decimals
    to_percentage= (nr_of_pixels_for_each_label*numpy.ones([len(average_confusion_matrix),len(average_confusion_matrix)])).transpose()/100
    average_confusion_matrix_percentages= numpy.round(average_confusion_matrix/to_percentage,2)





    #MOVED TO separate variable statistics_dictionary["average_confusion_matrix"]=json.dumps(average_confusion_matrix.tolist())

    #print(average_confusion_matrix)
    #print(sum(average_confusion_matrix.flatten()))
    imsisize= [1000,1000]
    try:
        assert(int(round(sum(average_confusion_matrix.flatten()))) == imsisize[0]*imsisize[1])
    except:
        print("WARNING, sum(average_confusion_matrix.flatten())"+str(sum(average_confusion_matrix.flatten())))
        print("imsisize[0]*imsisize[1]:"+str(imsisize[0]*imsisize[1]))

    #sum up all the values for the selected columns
    columns_to_sum_names=[column_name for column_name in pandas_dataframe.columns if not column_name in ['filename','confusionmatrix']]#sum error rates and iuo
    for n_column in range(len(columns_to_sum_names)):
        column_name= columns_to_sum_names[n_column]
        statistics_dictionary[column_name] = pandas_dataframe[column_name].sum()/number_of_images



    return (statistics_dictionary,average_confusion_matrix_percentages)

def get_bolean_index_array_for_filtering_dataset(row,subset):
    "return True if the file is in the relevant subset of files "
    
    file_name=row["filename"]
    #input("image_file_path:"+str(image_file_path))

    #input(" pathlib.Path(image_file_path):"+str( pathlib.Path(image_file_path)))
    #input(" pathlib.Path(image_file_path).parents:"+str( [a for a in pathlib.Path(image_file_path).parents]))

    #file_name = pathlib.Path(image_file_path).name.replace("\\","/").replace("\\","/").split("/")[-1] # os.path.split(image_file_path)[-1]

    #input(file_name)
    #input(subset)



    files_in_dataset_that_matches_the_subset_file_name =[file_name  for  subset_filename in subset if file_name in subset_filename]

    

    if len(files_in_dataset_that_matches_the_subset_file_name) ==0:

        return False
    else:

        #input("TRUE!")

        return True





def create_reports(dataset,pandas_dataframe,predictions_dir,subset_file,path_to_labels,report_cache_directory,model_performance_on_each_image_csv_root,experiment_settings_dict):
    """
    @arg pandas_dataframe: a dataframe that only includes information about the images included in a specific_subset
    make pdf reports on the given subset of the dataset
    create statistics about the specific subset
    """

    number_of_images = pandas_dataframe.shape[0]
    #print("dataframe dimensions: " + str(number_of_images))

    data_top = pandas_dataframe.head()
    #print("column names : " + str(data_top))

    #print(sorted)
    
    (statistics_dictionary, average_confusion_matrix)= create_statistics(pandas_dataframe=pandas_dataframe )
    statistics_dictionary["subset"]=subset_file
    subset_name = pathlib.Path(subset_file).stem

    #make sure the report cache directory exists
    os.makedirs(report_cache_directory, exist_ok = True)

    #save the statistics csv file in the report cache directory
    csv_statistics_file_path = report_cache_directory / pathlib.Path(subset_name+"_statistics.csv")
    save_statistics(statistics_dictionary,csv_file_name= csv_statistics_file_path) #str(subset_file).rstrip(".txt")+"_statistics.csv" )

    csv_confusion_matrix_file_path = report_cache_directory / pathlib.Path(subset_name+"_confusionmatrix.csv")
    #save confusio matrix in the report cache directory
    save_confusion_matrix(average_confusion_matrix,path_to_labels,csv_confusion_matrix_file_path,experiment_settings_dict) #str(subset_file).rstrip(".txt")+"_confusionmatrix.csv")
    

    nr_of_images_to_show = experiment_settings_dict["nr_of_images_to_show"]# 4 # 20
    #1. worst error rate
    #1.1 locate the paths to the images, predictions and labels
    worst_predicted_images_paths_list=get_image_paths(pandas_dataframe=pandas_dataframe,predictions_dir=predictions_dir,test_lb_dir=path_to_labels, nr_of_images_to_show=nr_of_images_to_show,
                  property_to_sort_on="error_rate", ascending=False, index_of_first_sorted_image_to_show=0,model_performance_on_each_image_csv_root=model_performance_on_each_image_csv_root,experiment_settings_dict=experiment_settings_dict)



    subset_name = subset_file.name
    

    
    #1.2create pdf report
    create_pdf_report(dataset,worst_predicted_images_paths_list,nr_of_images_to_show=nr_of_images_to_show,
                  property_to_sort_on="error_rate", ascending=False,
                  index_of_first_sorted_image_to_show=0,subset_name=subset_name,report_cache_directory=report_cache_directory)
    #2 best error rate
    #2.1 locate the paths to the images, predictions and labels
    best_predicted_images_paths_list = get_image_paths(pandas_dataframe=pandas_dataframe,
                                                        predictions_dir=predictions_dir, test_lb_dir=path_to_labels,
                                                        nr_of_images_to_show=nr_of_images_to_show,
                                                        property_to_sort_on="error_rate", ascending=True,
                                                        index_of_first_sorted_image_to_show=0,model_performance_on_each_image_csv_root=model_performance_on_each_image_csv_root,experiment_settings_dict=experiment_settings_dict)
    #2.2 create pdf report
    create_pdf_report(dataset,best_predicted_images_paths_list, nr_of_images_to_show=nr_of_images_to_show,
                      property_to_sort_on="error_rate", ascending=True,
                      index_of_first_sorted_image_to_show=0, subset_name=subset_name,report_cache_directory=report_cache_directory)
    #3 average error rate
    #3.1 locate the paths to the images, predictions and labels
    index_of_first_sorted_image_to_show = int(((number_of_images / 2) - nr_of_images_to_show / 2))

    
    average_predicted_images_paths_list = get_image_paths(pandas_dataframe=pandas_dataframe,
                                                        predictions_dir=predictions_dir, test_lb_dir=path_to_labels,
                                                        nr_of_images_to_show=nr_of_images_to_show,
                                                        property_to_sort_on="error_rate", ascending=True,
                                                        index_of_first_sorted_image_to_show=index_of_first_sorted_image_to_show,model_performance_on_each_image_csv_root=model_performance_on_each_image_csv_root,experiment_settings_dict=experiment_settings_dict)
    #3.2 create pdf report
    create_pdf_report(dataset,average_predicted_images_paths_list, nr_of_images_to_show=nr_of_images_to_show,
                      property_to_sort_on="error_rate", ascending=True,
                      index_of_first_sorted_image_to_show=index_of_first_sorted_image_to_show, subset_name=subset_name,report_cache_directory=report_cache_directory,average_images = True)



    return statistics_dictionary


"""
DEPRICTATED
def visualize_statistics(statistics_subsets):
    #Visualize how much of the mistakes come from the different subsets of images
    
    label_names = ["error_rate_summed_over_all_images","average_error_rate","average_iou_building"]
    bottom= numpy.array([0 for _ in label_names])
    fig, ax = plt.subplots()
    for subset in statistics_subsets:
        ax.bar(label_names,[subset[label_name] for label_name in label_names], width=0.35, label=subset["subset"],bottom= list(bottom))


        bottom=bottom+ numpy.array([subset[label_name] for label_name in label_names]) #increase the bottom so the bar starts at a higher position on next subset
    plt.xticks(rotation=7)
    ax.set_title("statistics")
    ax.legend()
    #plt.show()
    statistics_file_namne= "_".join([subset["subset"] for subset in statistics_subsets])+"_statistics.png"
    fig.savefig(statistics_file_namne)
"""

#def main(model_fn,test_set,subsets,test_set_dir,predictions_dir,path_to_labels):
def main(experiment_settings_dict):
    """
    create pdf visualisations of the different subsets 
    create pickle files with information about the images in the pdf files

    create csv files with statistics for each subset
    """
   


    #if os.path.exists(test_set_dir /'labelsByg/masks'):
    #    path_to_labels = test_set_dir /'labelsByg/masks'
    #else:
    #    path_to_labels = test_set_dir /'labels/masks'
    #    assert os.path.exists(path_to_labels), print("file does not exist! "+str(path_to_labels))


   

    #read in the erorr rates etc that we got when running
    
    csv_file_name = experiment_settings_dict["model_performance_on_each_image_csv"]
    #input(csv_file_name)
    pandas_dataframe = pd.read_csv(csv_file_name,sep=";")


    

    #statistics_subsets = []
    print(experiment_settings_dict["paths_to_subset_files"])

    #for each subset
    for subset_file in experiment_settings_dict["paths_to_subset_files"]:
 

        subset_files = get_files_in_subset(subset_file)

        
        #input(pandas_dataframe)
        #create an index array into the rows in the pandas dataframe
        #there is a 1 for each file that is part of the subset
        #there is a 0 for each file that not is part of the subset
        boolean_array_to_filtered_pandas_dataframe = pandas_dataframe.apply(
            get_bolean_index_array_for_filtering_dataset, axis=1, args=(subset_files,))


        print("sum(boolean_array_to_filtered_pandas_dataframe.to_numpy()) : " + str(
            sum(boolean_array_to_filtered_pandas_dataframe.to_numpy())))
        if (sum(boolean_array_to_filtered_pandas_dataframe.to_numpy()) ==0):
            sys.exit("rapport.py says : no files in subset : ")
 
        #filtered_dataframe is a pandas dataframe that only contains informtion about a special subset
        filtered_dataframe = pandas_dataframe[boolean_array_to_filtered_pandas_dataframe]
        

        statistics_subset = create_reports(experiment_settings_dict["dataset_folder"],filtered_dataframe,predictions_dir=experiment_settings_dict["prediction_folder"], subset_file=subset_file,path_to_labels=experiment_settings_dict["path_to_labels"],report_cache_directory = experiment_settings_dict["report_cache_directory"],model_performance_on_each_image_csv_root=experiment_settings_dict["path_to_images"],experiment_settings_dict=experiment_settings_dict)
        #statistics_subsets.append(statistics_subset)

    #visualize_statistics(statistics_subsets)










if __name__ == '__main__':
    roots = determine_root_dirs()
    print("roots : "+str(roots))
    
    # Styringsparametre:
    #model = 'mortensfavoritmodel'
    #bruger en pkl version av mortens favorit model /mnt/models/mortensfavoritmodel/RoofTopRGB1000p_Byg_Iteration27_1_step1_epoch_21.pkl
    #model_fn = 'RoofTopRGB1000p_Byg_Iteration27_1_step1_epoch_21.pkl'



    model = 'VtransEsbjergplus'  # 'mortensfavoritmodel'

    # bruger en pkl version av mortens favorit model /mnt/models/mortensfavoritmodel/RoofTopRGB1000p_Byg_Iteration27_1_step1_epoch_21.pkl
    model_fn = 'VtransEsbjergplus_step1_epoch_9.pth'  # 'RoofTopRGB1000p_Byg_Iteration27_1_step1_epoch_21.pkl'



    test_set = 'AmagerLangeland_Raabilleder'
    model_file = roots.models_root_dir()/model/model_fn
    test_set_dir = roots.train_root_dir()/test_set
    predictions_dir = roots.models_root_dir()/model/(model_fn + '_test_' + test_set)
    path_to_labels = test_set_dir /'labelsByg/masks'
    # subsets = ["land.txt", "bykerne.txt", "forstad.txt"]
    subsets = ["bykerne.txt"]
    os.system("pause")
    main(model_file,  subsets, test_set_dir)









    #main(model_fn,test_set,subsets,test_set_dir,predictions_dir,path_to_labels)
















