import numpy as np
import shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # to enable opening of large images
import pathlib
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
import rasterio
import matplotlib.patches as mpatches

#colors used when showing for what pixels the label matches the prediction
yellow=[255,255,0]
red=[255,0,0]
blue=[0,0,255]
green=[0,255,0]
orange=[255,165,0]
cyan=[0,255,255]
pink=[255,96,208]
white=[0,0,0]

# make a lookup table for accessing color for each label,prediction combination
max_nr_of_labels=15
# all cells with label != background ,and prediction != label should be yellow
# start by setting everything to red (yellow is to hard to se)
#NOTE: in some older implementations these were colored CYAN in order to tell them from label != backround && prediction == background
confusion_matrix_colors =np.array([[cyan for i in range(max_nr_of_labels)] for i in range(max_nr_of_labels)])
# all cells with label==ignore should blue if not predictions == background
confusion_matrix_colors[0,1:]=blue
# label == background AND prediction == background ,should be white
confusion_matrix_colors[1,1]=white
# all cells with label == predictions should be green
for i in range(1,max_nr_of_labels):
    confusion_matrix_colors[i,i]=green
# all cells with label ==background AND prediction!= background ,should be red
confusion_matrix_colors[1,2:]=red
# all cells with label != background AND prediction =background , should be pink
confusion_matrix_colors[2:,1]=pink

"""

labels_colors={1:{"name":"building","true_color":green,"false_positive":blue,"false_negative":red},\
               5:{"name":"Skov","true_color":green,"false_positive":blue,"false_negative":red},\
               2:{"name":"Vej","true_color":green,"false_positive":blue,"false_negative":red},\
               4:{"name":"Vand","true_color":green,"false_positive":blue,"false_negative":red},\
               3:{"name":"Mark","true_color":green,"false_positive":blue,"false_negative":red}}

"""


def create_concatenated_image(im1,im2):
    """
    gør en enkel bild bestående af två enskilde bilieder, side by side (e.g en rgb bilede og en maske-bilede)
    """
    assert(im1.width == im2.width)
    assert (im1.height == im2.height)
    concatenated_image = Image.new('RGBA', (im1.width+im2.width, im1.height ))
    concatenated_image.paste(im1, (0, 0))
    concatenated_image.paste(im2, (im1.width,0))
    return concatenated_image

def image_from_mask(mask,color=[0,0,255]):
    RGB_array = np.zeros((mask.shape[0], mask.shape[1], 3),
                                 dtype=np.uint8)
    # Make pixels visible
    RGB_array[mask] = color
    return Image.fromarray(RGB_array)

def add_Achannel(np_im,opaqenes):
    np_im_mask = (np_im.sum(axis=-1)>0)
    np_im_mask=np.expand_dims(np_im_mask,axis=-1)

    # opaqenes in range [0,255] 255 ==opaqe)


    with_Achannel = np.ones((np_im.shape[0], np_im.shape[1] ,np_im.shape[2]+1),dtype=np.uint8)*opaqenes #creating a opaqeness filled array with an extra chanell for the opaqeness
    with_Achannel=with_Achannel*np_im_mask # multiply the mask with the opaqeness value , so the 0 values in teh mask becomes 0 and the 1 values become == opaqeness

    with_Achannel[:,:, :-1] = np_im
    return with_Achannel
'''
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
'''



def visualize_image_prediction_and_label(image_path,label_path,prediction_path,show,save,visualization_name,class_remappings,visualization_text= "\n  hover with mouse over label or prediction image-position to get value",afterburn_path=None,save_prediction=False,names=[]):
    """
    #combine all images to a single visualisation
    :param image_path:
    :param label_path:
    :param prediction_path:
    :param show: should we show the visualization? true /false
    :param visualization_text: text describing what the different colors/values represent e.g :  'background:0,roads:1 , buildings:2'
    :param save_prediction: do we want to save the prediction image as a separate image?
    :return: image prediction and label as a single image
    """

    numpy_label= np.squeeze(np.array(Image.open(label_path),dtype=np.uint8))
    try:
        numpy_pred= np.squeeze(np.array(Image.open(prediction_path),dtype=np.uint8))
    except:
        #if probs have been saved instead of the argmax we need to do argmax
        pred_mask_img = rasterio.open(prediction_path).read()
        numpy_pred = np.array(pred_mask_img,dtype=float).argmax(axis=0).astype(np.uint8)

    if class_remappings != {}:
        (numpy_label,numpy_pred) = remap_classes(numpy_label,numpy_pred,class_remappings)




    #buildings and other known entities can safely be burned into the output
    building_color = 6

    if afterburn_path:
        burned_mask = np.squeeze(np.array(Image.open(afterburn_path)))!=0
        burned_value = building_color
        numpy_label[burned_mask]= burned_value
        numpy_pred[burned_mask] = burned_value



    #for some reason (most probably becaus we also render a RGB image )the vmin and vmax values dont work.
    # As a hack I set two pixels to the max and min values and force the visualization to use 0 as min value that way.
    numpy_pred[0,0]=0
    numpy_pred[-1,-1]=10
    numpy_label[0,0]=0
    numpy_label[-1,-1]=10










    #to visualize several images we create a matplotlib.pyplot figure
    fig = plt.figure()


    #Use grid spec so we can use differetn numbers of columns on different rows
    #Right nowwe only use a single row
    columns = 4
    rows = 1
    gs = GridSpec(rows, columns, figure=fig)

    #first image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(image_path.name,fontsize=5)  # set title
    plt.imshow(Image.open(image_path).convert("RGB")) #renders the image to the subplot


    #ax1.

    #second image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("label mask")  # set title
    plt.imshow(numpy_label,cmap="tab10",vmin=0,interpolation='nearest', vmax=10) #renders the label to the subplot
    #plt.colorbar(ax=[ax2],use_gridspec=True,ticks=[0,1,2,3,4,5]);

    #third image
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("prediction mask")  # set title
    plt.imshow(numpy_pred,cmap="tab10",vmin=0,interpolation='nearest', vmax=10) #renders the prediction to the subplot
    #plt.colorbar(ax=[ax3],use_gridspec=True,ticks=[0,1,2,3,4,5]);

    if save_prediction:
        plt.imsave("prediction_"+str(pathlib.Path(visualization_name).with_suffix(".jpg")),numpy_pred,cmap="tab10",vmin=0, vmax=10)

    #fourth image
    label_predictionion_match_visualization = get_prediction_label_match_visualization(label_path,prediction_path,class_remappings=class_remappings,verbose=False)
    #buildings and other known entities can safely be burned into the output
    if afterburn_path:
        burned_mask = np.array(Image.open(afterburn_path))!=0
        cmap = matplotlib.cm.get_cmap('tab10')
        rgba = cmap(building_color)
        label_predictionion_match_visualization[burned_mask] = rgba[0:3]



    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_title("match")  # set title
    plt.imshow(np.array(label_predictionion_match_visualization,dtype=np.uint8),cmap="tab10",interpolation='nearest',vmin=0, vmax=10) #renders the prediction to the subplot

    legend_axis = ax3
    #ad a legend that shows what the colors mean
    plt.rc('legend',fontsize=6)
    cmap = matplotlib.cm.get_cmap('tab10')



    #names= ["unknown","asfalt","fliser","grus","ubefestet","green_roof","drivhus","betonflade","brosten","unknown2","solceller"]
    patches=[mpatches.Patch(color=cmap(i/10), label=names[i]) for i in range(len(names))]





    legend_axis.legend(handles=patches,bbox_to_anchor=(1.05, 1),
            loc='upper left', borderaxespad=0.)



    #remove empty space betwen the subplots
    plt.tight_layout()

    #bbox_inches='tight',pad_inches = 0 removes
    #dpi=sets the reolution of the image
    fig.savefig("tmp.png",bbox_inches='tight',pad_inches = 0,dpi=300)


    if save:
        shutil.copyfile("tmp.png", visualization_name)
        print("saved :"+visualization_name)
        #fig.savefig(visualization_name)
    if show:
        plt.suptitle("visualisation of image: "+image_path.name + visualization_text)
        #set title of the main figure
        plt.show() #show to screen






    return Image.open("tmp.png").convert("RGB")


def image_from_masks(label_mask, pred_mask,false_negative_color=[255,0,0],false_positive_color=[0,0,255],true_positive_color=[0,255,0]):
    """
    visualize a label and prediction mask as a colorized image

    """
    RGB_array = np.zeros((label_mask.shape[0], label_mask.shape[1], 3),
                            dtype=np.uint8)
    #find intersection
    intersection_mask=label_mask*pred_mask
    intersection_mask_color=true_positive_color
    #remove intersection from masks
    label_mask= label_mask*np.invert(intersection_mask)
    pred_mask = pred_mask * np.invert(intersection_mask)

    # Make pixels visible
    RGB_array[label_mask] = false_negative_color
    RGB_array[pred_mask] = false_positive_color
    RGB_array[intersection_mask] = intersection_mask_color

    return Image.fromarray(RGB_array)

def image_from_image_and_mask(image,RGBmask):
    image = image.convert('RGBA')

    mask_RGBA_im = Image.fromarray(add_Achannel(RGBmask, opaqenes=100))


    

    composite_image = Image.alpha_composite(image, mask_RGBA_im)

    return composite_image

def remap_classes(numpy_label,numpy_pred,class_remappings):
    """
    @ arg numpy_label array
    @ arg numpy_pred array
    @arg class_remappings: a dictinary defining what classes should be mapped to other classes e.g {3:2,5:2,7:0} "change all class = 3 and class= 5 to 2, change all class 7 to 0"

    @ return (numpy_label,numpy_pred) the remapped label and prediction classes
    """

    # we first make sure that all classes are in the correct format
    class_remappings= {int(k):int(v) for k,v in class_remappings.items()}
    #if a mapping is given we first remap all affected classes
    for old_class in class_remappings:
        # remap label

        # find the pixels that should be remapped
        old_class_pixels=numpy_label==old_class

        # remap the classes
        numpy_label[old_class_pixels]=class_remappings[old_class]

        # remap prediction

        # find the pixels that should be remapped
        old_class_pixels=numpy_pred==old_class

        # remap the classes
        numpy_pred[old_class_pixels]=class_remappings[old_class]

    return (np.array(numpy_label,dtype=np.uint8),np.array(numpy_pred,dtype=np.uint8))


def get_prediction_label_match_visualization(label_path,prediction_path,class_remappings={},verbose=False):
    """
    :param label_path:
    :param prediction_path:
    :param class_remappings: a dictinary defining what classes should be mapped to other classes e.g {3:2,5:2,7:0} "change all class = 3 and class= 5 to 2, change all class 7 to 0"
    :param verbose:
    :return: visualization of the classification and label

    Can handle many different label categories
    #colors reprents how the predictions compare to the label ontop. (green ==(label ==prediction) (transparant ==(label==prediction==background)), red==(prediction==background AND label != background)), cyan ==(label!=prediction!=background)
    """
    if verbose:
        print("coloring")
        print("label_path:"+str(label_path))
        print("prediction_path:"+str(prediction_path))
    im = Image.open(label_path)

    numpy_label=np.array(im,dtype=np.uint8)
    if verbose:
        print("number of non_zero label_pixels : "+str((numpy_label>0).flatten().sum()))

    if prediction_path:
        try:
            numpy_pred = np.array(Image.open(prediction_path),dtype=np.uint8)
        except:
            #if probs have been saved instead of the argmax we need to do argmax
            pred_mask_img = rasterio.open(prediction_path).read()
            numpy_pred = np.array(pred_mask_img,dtype=float).argmax(axis=0).astype(np.uint8)

    else:
        numpy_pred = numpy_label

    #should some classes be remapped to other classes
    if class_remappings != {}:
        (numpy_label,numpy_pred) = remap_classes(numpy_label,numpy_pred,class_remappings)

    def get_color(label_and_prediction):
        """
        arg: label_and_prediction: a list of two integers , [label,prediction]
        returns the color that is used for visualizing the label-prediction combination
        """
        assert len(label_and_prediction) ==2
        label=int(label_and_prediction[0])
        prediction = int(label_and_prediction[1])
        if 255 in label_and_prediction:
            #print("coloring 255==IGNORE")
            return cyan
        else:
            
            return confusion_matrix_colors[label,prediction]
    if verbose:
        print("numpy_label.shape:"+str(numpy_label.shape))
        print("numpy_pred.shape:"+str(numpy_pred.shape))



    label_predictions_pairs= np.stack([numpy_label,numpy_pred],axis=2)

    # This should be replaced with indexing_arrays! e.g flatten label_predictions_pairs to format [pair1,pair2,pair3.. pair_n] ,and trasnlate the 2dimensional indexing to 1dim indexing (there is a function for this)
    colored_image = np.array(np.apply_along_axis(get_color,axis=2,arr=label_predictions_pairs))
    return colored_image

def masked_image_from_image_prediction_label(image_path,label_path,prediction_path,building_path,class_remappings={},verbose=False,save=False,show=False,visualization_name="visualized",afterburn_path = None):
    """
    alternative to visualize_triplet() they should most probably merge eventually

    given paths to image, label and prediction,
    creates an image with original to the left and a visualization of the classification and label to the right.
    Can handle many different label categories
    #colors reprents how the predictions compare to the label ontop. (green ==(label ==prediction) (transparant ==(label==prediction==background)), red==(prediction==background AND label != background)), cyan ==(label!=prediction!=background)
    @arg class_remappings: a dictinary defining what classes should be mapped to other classes e.g {3:2,5:2,7:0} "change all class = 3 and class= 5 to 2, change all class 7 to 0"
    @arg afterburn_path: path to folder with image-data that should be burned in  on the resulting image
    """

    if verbose:
        print("coloring")
        print("image_path:"+str(image_path))
        print("label_path:"+str(label_path))
        print("prediction_path:"+str(prediction_path))
    im = Image.open(label_path)
    
    numpy_label=np.array(im,dtype=np.uint8)
    if building_path!= None:
        try:
            print("ad the bulding masks to the labels")
            bulding_numpy = np.array(Image.open(building_path),dtype=np.uint8)
            where_there_are_buildings = bulding_numpy!=0
            numpy_label[where_there_are_buildings] = bulding_numpy[where_there_are_buildings]
        except:
            print("failed to ad buldings")
    if verbose:
        print("number of non_zero label_pixels : "+str((numpy_label>0).flatten().sum()))
    input_image = Image.open(image_path)
    if prediction_path:
        try:
            numpy_pred = np.array(Image.open(prediction_path),dtype=np.uint8)
        except:
            #if probs have been saved instead of the argmax we need to do argmax
            pred_mask_img = rasterio.open(prediction_path).read()
            numpy_pred = np.array(pred_mask_img,dtype=float).argmax(axis=0).astype(np.uint8)
    else:
        numpy_pred = numpy_label

    #should some classes be remapped to other classes     
    if class_remappings != {}:
        (numpy_label,numpy_pred) = remap_classes(numpy_label,numpy_pred,class_remappings)

    def get_color(label_and_prediction):
        """
        arg: label_and_prediction: a list of two integers , [label,prediction] 
        returns the color that is used for visualizing the label-prediction combination
        """
        assert len(label_and_prediction) ==2
        label=label_and_prediction[0]
        prediction = label_and_prediction[1]
        if 255 in label_and_prediction:
            #print("coloring 255==IGNORE")
            return cyan
        else:
            return confusion_matrix_colors[label,prediction]

    if verbose:
        print("numpy_label.shape:"+str(numpy_label.shape))
        print("numpy_pred.shape:"+str(numpy_pred.shape))

    label_predictions_pairs= np.stack([numpy_label,numpy_pred],axis=2)

    # This should be replaced with indexing_arrays! e.g flatten label_predictions_pairs to format [pair1,pair2,pair3.. pair_n] ,and trasnlate the 2dimensional indexing to 1dim indexing (there is a function for this)
    colored_image = np.array(np.apply_along_axis(get_color,axis=2,arr=label_predictions_pairs))
    #should_be_trasnparant=colors==white

    #buildings and other known entities can safely be burned into the output
    if afterburn_path:
        burned_mask = np.array(Image.open(afterburn_path))!=0
        burned_value = [255,255,255]
        colored_image[burned_mask] = burned_value


    composite_image = image_from_image_and_mask(input_image, colored_image)






    original_and_masked_images = create_concatenated_image(input_image, composite_image)



    if save:

        visualization_name=visualization_name+pathlib.Path(image_path).name
        original_and_masked_images.save(visualization_name)


        print("saved :"+visualization_name)
        #fig.savefig(visualization_name)
    if show:
        original_and_masked_images.show()




    return original_and_masked_images


def visualize_triplet(image_path,label_path,prediction_path,save,show,class_remappings,afterburn_path = None,save_prediction=False,names=["unknown","asfalt","fliser","grus","ubefestet","green_roof","drivhus","betonflade","brosten","unknown2","solceller"]):
    """
    Visualize how prediction and label matches together with image,

    :param image_path: path/to/image or folder
    :param label_path: path/to/label or folder
    :param prediction_path: path/to/prediction or folder
    :param class_remappings: e.g {255:0}
    :return:
    """
    if os.path.isdir(image_path):
        image_dir = image_path
        label_dir = label_path
        prediction_dir = prediction_path
        #if all paths are directories we visualize all images in the folders
        assert os.path.isdir(image_dir) and os.path.isdir(label_path) and os.path.isdir(prediction_path)
        filenames = [pathlib.Path(filename) for filename in os.listdir(image_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff','.tif', '.bmp', '.gif'))]

        for filename in filenames:
            image_path,label_path,prediction_path = image_dir/pathlib.Path(filename) ,label_dir/pathlib.Path(filename),prediction_dir/pathlib.Path(filename)


            #create  a matplot lib figure with image label prediction and visualization. show to screen i show=True, return a pil image
            visualize_image_prediction_and_label(image_path,label_path,prediction_path,show=show,save=save,visualization_name="visualized"+image_path.name,class_remappings=class_remappings,afterburn_path=afterburn_path,save_prediction=save_prediction,names=names)
            '''
            if save:
                print("saving :"+"visualized"+image_path.name)
                image_label_prediction_visualization.save("visualized"+image_path.name)
            '''


            #im = masked_image_from_image_prediction_label(image_path=image_path,label_path=label_path,prediction_path=prediction_path,class_remappings={})
            #if save:
            #    im.save("visualized_"+filename.name)
            #if show:
            #    im.show()
        #If the paths point to images we visualize only teh relevant image

    else:
        visualize_image_prediction_and_label(image_path,label_path,prediction_path,show=show,save=save,visualization_name="visualized"+image_path.name,class_remappings=class_remappings,afterburn_path=afterburn_path,save_prediction=save_prediction,names=names)
        '''

        im = masked_image_from_image_prediction_label(image_path=image_path,label_path=label_path,prediction_path=prediction_path,class_remappings={})

        if save:
            im.save("visualized"+pathlib.Path(args.Image).name)
        if show:
            im.show()
        '''

if __name__ == "__main__":
    example_usage= r"python image_from_mask.py -i F:\SDFE\DATA\OrtofotoDeepLearning\trainingdata\AmagerLangelandRaabilleder\images\2020_83_39_13_3222_0_9000.tif -l F:\SDFE\DATA\OrtofotoDeepLearning\trainingdata\AmagerLangelandRaabilleder\labelsByg\masks\2020_83_39_13_3222_0_9000.tif"
    print("########################EXAMPLE USAGE########################")
    print(example_usage)
    print("#############################################################")
    import argparse


    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--Image", help="path/to/image.png  or path/to/folder",required=True)
    parser.add_argument("-p", "--Prediction", help="path/to/predictionmask.png  or path/to/folder",required=False)
    parser.add_argument("-l", "--Label",  help="path/to/labelmask.png  or path/to/folder",required=True)
    parser.add_argument('-s','--Save',
                        help='should we save the image to disk -s-> True , no -s -> False: ',action = 'store_true',default=False)
    parser.add_argument('-v','--Show',
                        help='should we show the image to screen? -v-> True , no -s -> False: ',action = 'store_true',default=False)
    parser.add_argument("-a", "--AfterBurnFolder", help="path/to/folder  e.g path/to/knownhouses",required=False)
    parser.add_argument('--save_prediction',
                        help='should we save prediction in separate imgage? --save_prediction --> True , no --save_prediction -> False: ', action='store_true',
                        default=False)

    args = parser.parse_args()


    image_path = pathlib.Path(args.Image)
    label_path = pathlib.Path(args.Label)
    prediction_path = pathlib.Path(args.Prediction)
    if args.AfterBurnFolder:
        AfterBurnFolder = pathlib.Path(args.AfterBurnFolder)/image_path.name
    else:
        AfterBurnFolder = None


    visualize_triplet(image_path,label_path,prediction_path,save = args.Save,afterburn_path=AfterBurnFolder,show= args.Show,class_remappings={},save_prediction=args.save_prediction,names=["unknown","asfalt","fliser","grus","ubefestet","green_roof","drivhus","betonflade","brosten","unknown2","solceller"])


