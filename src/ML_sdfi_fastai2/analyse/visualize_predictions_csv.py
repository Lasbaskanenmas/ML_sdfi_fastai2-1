import numpy as np
import json
import argparse
import pandas as pd
from PIL import Image
import pathlib
import image_from_mask
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mplcursors


# Define the function to format the displayed information
def hover_prediction_formatter(sel):
    x, y = int(sel.target[0]), int(sel.target[1])
    sel.annotation.set_text(f'\nPrediction: {class_names[int(prediction_numpy[y,x])]}')

def hover_label_formatter(sel):
    x, y = int(sel.target[0]), int(sel.target[1])
    sel.annotation.set_text(f'\nLabel: {class_names[int(label_numpy[y,x])]}')

if __name__ == "__main__":
    # sorts the files in csv acording to erro rrate
    #visualize the label ,prediction and where in the image the precdiction matches with the label

    example_usage1= "python visualize_predictions_csv.py -c \\TRUENAS\mlnas\mnt\logs_and_models\downweighted_forest\downweighted_forest\logs\model_performance_on_each_image.csv -l T:\mnt\trainingdata\landsdaekkende_dataset\LandsdaekkendeTest2021RGBN1000pRAWNaNBygVejVandSkov\labels\masks -p T:\mnt\logs_and_models\downweighted_forest\downweighted_forest\models\Rasmus_downweighted_forest\LandsdaekkendeTest2021RGBN1000pRAWNaNBygVejVandSkov --Images \\TRUENAS\mlnas\mnt\trainingdata\landsdaekkende_dataset\LandsdaekkendeTest2021RGBN1000pRAWNaNBygVejVandSkov\images --Codes path\to\codes.txt"
    example_usage2= "python visualize_predictions_csv.py -c \\TRUENAS\mlnas\mnt\logs_and_models\downweighted_forest\downweighted_forest\logs\model_performance_on_each_image.csv -l T:\mnt\trainingdata\landsdaekkende_dataset\LandsdaekkendeTest2021RGBN1000pRAWNaNBygVejVandSkov\labels\masks -p T:\mnt\logs_and_models\downweighted_forest\downweighted_forest\models\Rasmus_downweighted_forest\LandsdaekkendeTest2021RGBN1000pRAWNaNBygVejVandSkov --Images \\TRUENAS\mlnas\mnt\trainingdata\landsdaekkende_dataset\LandsdaekkendeTest2021RGBN1000pRAWNaNBygVejVandSkov\images -r 2021_83_25_10_0420_5000_13000.tif --Codes path\to\codes.txt"

    print("########################EXAMPLE USAGE########################")
    print(example_usage1)
    print(example_usage2)
    print("#############################################################")


    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--Csv_file", help="path/to/model_on_dataet.csv,  a csv file with one row for each image a model has done inference on. rows for valid_loss valid_accuracy and train_loss ",required=True)

    parser.add_argument("-p", "--Predictions", help="path/to/folder_with_predictions",required=True)
    parser.add_argument("-i", "--Images", help="path/to/folder_with_images",required=True)
    parser.add_argument("-l", "--Labels",  help="path/to/folder_with_labels",required=True)
    parser.add_argument('-s','--Save',help='should we save the image to disk -s-> True , no -s -> False: ',action = 'store_true',default=True)
    parser.add_argument('-r','--Resume_from_Checked_File',help='should we skip to a particular file? . e.g  -c-> 2021_84_40_11_0703_7000_18000.tif  ',required=False)
    parser.add_argument('--Codes',help='e.g path/to/codes.txt',required =True)



    parser.add_argument("-t", "--threshold_ignore",  help="part of pixels that must not be of type ignore ==255.   e.g 0.5",required=False,default =0.0001,type=float)

    image_shape=[1000,1000]


    args = parser.parse_args()

    #parse the codes.txt file so we know what and how many classes to expect

    # Open the file in read mode
    with open(args.Codes, 'r') as file:
        # Iterate over each line
        # Remove any leading or trailing whitespace
        class_names = [line.strip() for line in file]
        class_numbers = range(len(class_names))







    pandas_dataframe= pd.read_csv(args.Csv_file,sep=";")
    
    
    #sort by error_rate so we get the images with bad  predictions first
    pandas_dataframe= pandas_dataframe.sort_values(by=['error_rate'],ascending=False)


    #if we want to continue from a checked filename we need to know if we seen it or not.
    seen_checked_filename = False
    
    for index, row in pandas_dataframe.iterrows():
        file_name =row['filename']
        image_path = pathlib.Path(args.Images)/pathlib.Path(file_name)
        label_path= pathlib.Path(args.Labels)/ pathlib.Path(file_name)
        prediction_path= pathlib.Path(args.Predictions)/ pathlib.Path(file_name)
        print("##")
        print("image_path")
        print( image_path)
        print("label_path")
        print( label_path)
        print("##")
        print("error_rate ")
        print(row['error_rate'])


        print("confusionmatrix")
        print(row['confusionmatrix'])
        print(json.loads(row['confusionmatrix']))
        number_of_not_ignored_pixels = np.array(json.loads(row['confusionmatrix'])).sum()
        if number_of_not_ignored_pixels > image_shape[0]*image_shape[1] *args.threshold_ignore:
            print("enough pixels have proper label_values (!=255(IGNORE))")

            #OLD TO BE REMOVED COMMENTS
            #The path to the NAS is differetn on windows and linux, we asume wewill be using windows
            #windows_version = row['filename'].replace("/mnt/T","T:")





            
            ##TO BE REMOVED COMMENT file_name = pathlib.Path(windows_version).name

            #if we want to continue from a checked filename we skip all files untill we encountered the filename
            if args.Resume_from_Checked_File and (not seen_checked_filename):
                if file_name == args.Resume_from_Checked_File:
                    seen_checked_filename = True
                else:
                    continue


            



          

            #Create an image with original image to the left , and original image with colors that reprents how the predictions compare to the label ontop. (green ==(label ==prediction) (transparant ==(label==prediction==background)), red==(prediction==background AND label != background)), cyan ==(label!=prediction!=background))
            image_and_visualization = image_from_mask.masked_image_from_image_prediction_label(image_path=image_path,label_path=label_path,prediction_path=prediction_path)


            #Create an image with label to the left, and predictions to the right
            #label_and_prediction = Image.open(label_path), Image.open(prediction_path))



            #combine all images to a single visualisation

            #to visualize several images we createa a matplotlib.pyplot figure
            fig = plt.figure(constrained_layout=True)


            #Use grid spec so we can use differetn numbers of columns on different rows
            columns = 2 
            rows = 2
            gs = GridSpec(rows, columns, figure=fig) 

            #first image shouold take the complete first row
            ax1 = fig.add_subplot(gs[0, :])
            ax1.set_title("original and label-prediction visualization")  # set title
            plt.imshow(image_and_visualization) #renders the image to the subplot


            #ax1.

            #second image shouold take the second row first column
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.set_title("label mask")  # set title
            label_numpy =np.array(Image.open(label_path),dtype=np.uint8)
            label= plt.imshow(label_numpy,cmap="tab20",vmin=0, vmax=len(class_names)) #renders the label to the subplot
            plt.colorbar(ax=[ax2],use_gridspec=True,ticks=class_numbers);

            #third image shouold take the second row second column
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.set_title("prediction mask")  # set title
            prediction_numpy = np.array(Image.open(prediction_path),dtype=np.uint8)
            prediction = plt.imshow(prediction_numpy,cmap="tab20",vmin=0, vmax=len(class_names)) #renders the prediction to the subplot
            plt.colorbar(ax=[ax3],use_gridspec=True,ticks=class_numbers);
      

            #set title of the main figure
            classes_description = "_".join([str(class_numbers[i])+ "=" +class_names[i] for i in range(len(class_names))])
            plt.suptitle("visualisation of image: "+image_path.name + "\n "+classes_description+"\n  hover with mouse over label or prediction image-position to get value")
            # Define the function to format the displayed information
            def formatter(**kwargs):
                # Customize the information displayed in the top right corner
                x = kwargs['x']
                y = kwargs['y']
                value = kwargs['z']
                class_name = class_names[int(value)]
                return f'X: {x:.2f}\nY: {y:.2f}\nValue: {value:.2f}' + "apa"

            # Configure the hover annotation using mplcursors
            #text = ax3.text(0.95, 0.95, '', transform=ax1.transAxes, ha='right', va='top')
            cursor = mplcursors.cursor(prediction,hover=True)
            cursor.connect("add", hover_prediction_formatter)

            cursor = mplcursors.cursor(label,hover=True)
            cursor.connect("add", hover_label_formatter)

            plt.show() #show to screen
        else:
            print("to many pixels are of type 255==IGNORE ")
            print("image :"+str(row['filename'])+ " is ignored ")
        
        
        



    """




    image_path = args.Image
    label_path = args.Label
    prediction_path = args.Prediction
    #prediction_path = r"\mnt\models\modeller_fra_ask\modeller\v2paaske_epoch_0\v2paaske_epoch_0.pkl_test_AmagerLangeland1000p160mmRTO\" + pathlib.Path(image_path).name
    #label_path = r"\mnt\trainingdata\AmagerLangeland1000p160mmRTO\labelsByg\masks\" +
    im = masked_image_from_image_prediction_label(image_path=image_path,label_path=label_path,prediction_path=prediction_path)
    #im = colored_image_from_image_prediction_and_label(image_path=image_path, label_path=label_path,prediction_path=prediction_path)
    if args.Save:
        im.save("visualized"+pathlib.Path(args.Image).name)
    im.show()
    """
