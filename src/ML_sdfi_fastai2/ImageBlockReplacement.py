from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.vision import *
import torch
from PIL import Image
import numpy as np
import rasterio
import pathlib
import sys
import json


"""
##############
#BACKGROUND  #
##############
When training we need to go from path_to_image to tensor, by opening image aply normalisations and transforms 
Some of these transforms also should be aplied to the masks
In fastai2 this is done in different kinds of transforms that are aplied in a certain order that can be hard to grasp

datapipeline = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files_function,
                   splitter=RandomSplitter(),
                   get_y=get_msk,
                   batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)])


ImageBlock provides two kind of transforms 
1. a type transform where it opens the path as PIL image
2. a batch transform that divides by 255                   

#This is the code for ImageBlock
ImageBlock(cls=PILImage):
    "A `TransformBlock` for images of `cls`"
    return TransformBlock(type_tfms=cls.create, batch_tfms=IntToFloatTensor)

notes:
    PILimage inherits from PILBASE
    Pilbase.create loads the PIL image form a path

Other transforms are listed in "batch_tfms"
1. where *aug_transforms() is a standard set of flips rotations etc
2. where Normalize.from_stats(*imagenet_stats) is done after theese transforms, and constitutes of a subtraction of mean values and division by std values


During training fastai aplies the differetn transforms in this order
1. ImageBlock.type_tfms  (open image as PIL image)
2. batch_tfms *aug_transforms() rotation and color changes etc
3. ImageBlock.batch_tfms (divide by 255)
4. batch_tfms Normalize.from_stats(*imagenet_stats) (subtrack mean and divide by std)

in order to do ImageBlock.batch_tfms between the two batch_tfms fastai use a ordering system
all transforms inherits from the Transform object. https://fastcore.fast.ai/transform.html
It has an "order" atribute wich is an integer that decides what should be done in what order (high order means early in the pipeline)
e.g: the following transform should be done before divison by 255 wich is done in  IntToFloatTensor (see below)
class LightingTfm(SpaceTfm):
    "Apply `fs` to the logits"
    order = 40
    
class IntToFloatTensor(DisplayedTransform):
    "Transform image to float tensor, optionally dividing by 255 (e.g. for images)."
    order = 10 #Need to run after PIL transforms on the GPU
    def __init__(self, div=255., div_mask=1): store_attr()
    def encodes(self, o:TensorImage): return o.float().div_(self.div)
    def encodes(self, o:TensorMask ): return o.long() // self.div_mask
    def decodes(self, o:TensorImage): return ((o.clamp(0., 1.) * self.div).long()) if self.div else o
    
    
##########################################################
#Implementing the same pipeline for multichannel images  #
##########################################################
The ImageBlock class can not handle more channels than 3 if it uses the default cls=PILImage class 
By creating a new class that implements the same functionality as PILImage 

"""
#old comments
'''
    

1. open the image as a PIL image (or ad potential extra channels and save result as a numpy array ): done in the Imageblock


def open(path)
    if we only use 3 channels we can represent this as a PILimage but with 5 channels we ned to use numpy or tensor format
    np_image = np.array(Image.open(path)) ++ extra channels
    np_image = np.array(np.vstack(np_image,new_channel.reshape([1]+list(new_channel.shape)))),dtype=np.float32)
2.
2.1 apply transforms, (batch_tfms: where *aug_transforms() is a standard set of flips rotations etc, adn , Normalize.from_stats(*imagenet_stats) is done after theese trasnforms, and constitutes of a subtraction of mean values and division by std values)
2.2 send the images through the "ImageBlock" object. (turn it into numpy(if in pilformat) and divide by 255) 

                   





                   
a_dataset = DataBlock(blocks= (TransformBlock(partial(open_img,experiment_settings_dict=experiment_settings_dict)), MaskBlock(codes)),get_items=sdfe_get_images, splitter=dataset_splitter,get_y= label_func ,item_tfms=sdfi_tfm,)                   







def open(path)
    if we only use 3 channels we can represent this as a PILimage but with 5 channels we ned to use numpy or tensor format
    np_image = np.array(Image.open(path)) ++ extra channels
    np_image = np.array(np.vstack(np_image,new_channel.reshape([1]+list(new_channel.shape)))),dtype=np.float32)
2.
2.1 apply transforms, (batch_tfms: where *aug_transforms() is a standard set of flips rotations etc, adn , Normalize.from_stats(*imagenet_stats) is done after theese trasnforms, and constitutes of a subtraction of mean values and division by std values)
2.2 send the images through the "ImageBlock" object. (turn it into numpy(if in pilformat) and divide by 255) 

camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files_function,
                   splitter=RandomSplitter(),
                   get_y=get_msk,
                   batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)])
                   
'''

def inspect_multichannel_image(multichannel_image_as_single_numpy_array):

    (channels, y, x) = multichannel_image_as_single_numpy_array.shape
    for n_channel, channel in enumerate(range(channels)):
        print("channel: " + str(n_channel))
        min = multichannel_image_as_single_numpy_array[channel].min()
        max = multichannel_image_as_single_numpy_array[channel].max()
        mean = multichannel_image_as_single_numpy_array[channel].mean()
        print(str([min, mean, max]))


        arr= np.array(255 * ((multichannel_image_as_single_numpy_array[channel] - min) / (max - min)),dtype=np.uint8)

        Image.fromarray(arr).save("channel_" + str(n_channel) + ".jpg")

        print("saved "+"channel_" + str(n_channel) + ".jpg")
    input("pres enter for saving the channels of the next image")

class MultiChannelBase(TensorImage):
    """
    A multi channel replacment for Pilbase
    meant to be sent to ImageBlock as replacment for PILImage like this
    ImageBlock(cls=PILImage)
    
    provides the create function that opens the filepath (filename/fn) and returns a MultiChannelImage object (inherits from MultiChannelBase wich inherits form torch.tensor)
    in other words this class provides a way of opening a file as a tensor
       
    
    """
    
    
    
    @classmethod
    def create(cls, fn:(Path,str,Tensor,ndarray,bytes), **kwargs)->None:
        "Open an `Image` from path `fn`"
        
        #the orginal fastai code as lots of ways of openign lots of differetn kind og datatypes (paths, numpy arrays tensors etc) for now we concentrate on opening a filename
        if isinstance(fn,np.ndarray):
            fn = fn

        else:

            #fn = np.array(load_image(fn, **kwargs))
            fn = load_all_datasources_for_image(fn,**kwargs)
            debug = False
            if debug:
                inspect_multichannel_image(fn)


        multi_channel_image =cls(fn)


        return multi_channel_image

    def show(self, ctx=None, **kwargs):
        "Show image using `merge(self._show_args, kwargs)`"
        return show_image(self, ctx=ctx, **merge(self._show_args, kwargs))

    def __repr__(self): return f'{self.__class__.__name__} mode={self.mode} size={"x".join([str(d) for d in self.shape])}'


# A multichannel replacment for PILImage
class MultiChannelImage(MultiChannelBase): pass


def MultichannelImageBlock(cls=MultiChannelImage,experiment_settings_dict=None):
    """A `TransformBlock` for images of `cls`
    A rewrite of fastais own Imageblock class, only difference is that we take a dictionary as extra argument in order to be able to send data from the dictionary into the MultiChannelImage.create class

    The original Imageblock applies the IntToFloatTensor transform (wich divades its inputs by 255) but for some reason this transform is for some reson aplied even if I dont ad it to the batch_tfms in this class
    Only solution I manged to find is to not ad it to batch_tfms in this class (we only want to to be aplied once) but I dont understund where it is added to the trasnforms
    TODO: find out where IntToFloatTensor is added to the list of transforms
    """
    #partial(a_function,an_argument) returns a function based on 'a_function' with an 'an_argument' as default value for one of the parameters
    type_tfms= partial(cls.create,experiment_settings_dict=experiment_settings_dict)
    return TransformBlock(type_tfms=type_tfms , batch_tfms=[])





def load_all_datasources_for_image(path,experiment_settings_dict):
    """
    :param path: path to image
    :param experiment_settings_dict: dictionary holding all info relevant for creating a multichannel image combined of several datatypes (e.g lidarvalues lodimages etc)
    """

    the_first = True
    for index, datatype in enumerate(experiment_settings_dict["datatypes"]):
        #input(datatype)
        #input(experiment_settings_dict["channels"])
        #input(path)
        #input(path.parent.parent/ pathlib.Path("splitted_"+datatype)/path.name)



        if the_first:
            the_first = False

            as_array = rasterio.open(path.parent.parent/ pathlib.Path(datatype)/path.name).read().astype('float32')
            as_array = as_array[experiment_settings_dict["channels"][index]]
        else:
            new_as_array = rasterio.open(path.parent.parent / pathlib.Path(datatype) / path.name).read().astype('float32')
            new_as_array = new_as_array[experiment_settings_dict["channels"][index]]
            as_array = np.vstack((as_array,new_as_array))
        #input(as_array.shape)
        #im=Image.open(path.parent.parent/ pathlib.Path("splitted_"+datatype)/path.name)
        #im.show()
        #input("ok?")
        #as_array= np.array(im)
        #print(len(experiment_settings_dict["channels"]))
        #print(experiment_settings_dict["channels"][index])
        #print(as_array.shape)

        #loaded_images.append(without_unwanted_channels)
    loaded_images_as_array= np.array(as_array,dtype=np.float32)


    return  loaded_images_as_array#loaded_images_as_array





