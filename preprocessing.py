# basic
import warnings
import os, gc, cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
# visualize
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

# segmentation tool
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei


# read and visualize sample image, only 3 channels
def read_rby_image(filename):

    '''
    read individual images
    of different filters (R, B, Y)
    and stack them.
    ---------------------------------
    Arguments:
    filename -- sample image path

    Returns:
    stacked_images -- stacked (RBY) image
    '''

    red = cv2.imread(os.path.join(ROOT, 'train/') + filename + "_red.png", cv2.IMREAD_UNCHANGED)
    blue = cv2.imread(os.path.join(ROOT, 'train/') + filename + "_blue.png", cv2.IMREAD_UNCHANGED)
    yellow = cv2.imread(os.path.join(ROOT, 'train/') + filename + "_yellow.png", cv2.IMREAD_UNCHANGED)

    stacked_images = np.transpose(np.array([red, blue, yellow]), (1,2,0))
    return stacked_images


# read and visualize sample image
def read_sample_image(filename):

    '''
    read individual images
    of different filters (R, G, B, Y)
    and stack them.
    ---------------------------------
    Arguments:
    filename -- sample image path

    Returns:
    stacked_images -- stacked (RGBY) image
    '''

    red = cv2.imread(os.path.join(ROOT, 'train/') + filename + "_red.png", cv2.IMREAD_UNCHANGED)
    green = cv2.imread(os.path.join(ROOT, 'train/') + filename + "_green.png", cv2.IMREAD_UNCHANGED)
    blue = cv2.imread(os.path.join(ROOT, 'train/') + filename + "_blue.png", cv2.IMREAD_UNCHANGED)
    yellow = cv2.imread(os.path.join(ROOT, 'train/') + filename + "_yellow.png", cv2.IMREAD_UNCHANGED)

    stacked_images = np.transpose(np.array([red, green, blue, yellow]), (1,2,0))
    return stacked_images


def plot_all(im, label):

    '''
    plot all RGBY image,
    Red, Green, Blue, Yellow,
    filters images.
    --------------------------
    Argument:
    im - image
    '''

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 5, 1)
    plt.imshow(im[:,:,:3])
    plt.title('RGBY Image')
    plt.axis('off')
    #plt.subplot(1, 5, 2)
    #plt.imshow(im[:,:,0], cmap='Reds')
    #plt.title('Microtubule channels')
    #plt.axis('off')
    #plt.subplot(1, 5, 3)
    #plt.imshow(im[:,:,1], cmap='Greens')
    #plt.title('Protein of Interest')
    #plt.axis('off')
    #plt.subplot(1, 5, 4)
    #plt.imshow(im[:,:,2], cmap='Blues')
    #plt.title('Nucleus')
    #plt.axis('off')
    #plt.subplot(1, 5, 5)
    #plt.imshow(im[:,:,3], cmap='Oranges')
    #plt.title('Endoplasmic Reticulum')
    #plt.axis('off')
    plt.show()


# read and visualize sample image
def read_sample_image_seg(filename):

    '''
    read individual images
    of different filters (R, B, Y)
    and stack them for segmentation.
    ---------------------------------
    Arguments:
    filename -- sample image file path

    Returns:
    stacked_images -- stacked (RBY) image path in lists.
    '''

    red = os.path.join(ROOT, 'train/') + filename + "_red.png"
    blue = os.path.join(ROOT, 'train/') + filename + "_blue.png"
    yellow = os.path.join(ROOT, 'train/') + filename + "_yellow.png"

    stacked_images = [[red], [yellow], [blue]]
    return stacked_images, red, blue, yellow


# segment cell
def segmentCell(image, segmentator, seg_with="yellow"):

    '''
    segment cell and nuclei from
    microtubules, endoplasmic reticulum,
    and nuclei (R, B, Y) filters.
    ------------------------------------
    Argument:
    image -- (R, B, Y) list of image arrays
    segmentator -- CellSegmentator class object

    Returns:
    cell_mask -- segmented cell mask
    '''
    mapping = {"red":0, "blue":1, "yellow":2}

    nuc_segmentations = segmentator.pred_nuclei(image[mapping[seg_with]])
    cell_segmentations = segmentator.pred_cells(image)
    nuclei_mask, cell_mask = label_cell(nuc_segmentations[0], cell_segmentations[0])

    gc.collect(); del nuc_segmentations; del cell_segmentations; del nuclei_mask

    return cell_mask


# plot segmented cells mask, image
def plot_cell_segments(mask, red, blue, yellow):

    '''
    plot segmented cells
    and images
    ---------------------
    Arguments:
    mask -- cell mask
    red -- red filter image path
    blue -- blue filter image path
    yellow -- yellow filter image path
    '''
    microtubule = plt.imread(r)
    endoplasmicrec = plt.imread(b)
    nuclei = plt.imread(y)
    img = np.dstack((microtubule, endoplasmicrec, nuclei))

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.6)
    plt.title('Image + Mask')
    plt.axis('off')
    plt.show()


# plot single segmented cells mask, image
def plot_single_cell(mask, red, blue, yellow):

    '''
    plot single cell mask
    and image
    ---------------------
    Arguments:
    mask -- cell mask
    red -- red filter image path
    blue -- blue filter image path
    yellow -- yellow filter image path
    '''
    microtubule = plt.imread(r)
    endoplasmicrec = plt.imread(b)
    nuclei = plt.imread(y)
    img = np.dstack((microtubule, endoplasmicrec, nuclei))

    contours= cv2.findContours(mask.astype('uint8'),
                               cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours[0]]
    x = np.argsort(areas)
    cnt = contours[0][x[-1]]
    x,yc,w,h = cv2.boundingRect(cnt)

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(img[yc:yc+h, x:x+w])
    plt.title('Cell Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask[yc:yc+h, x:x+w])
    plt.title('Cell Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img[yc:yc+h, x:x+w])
    plt.imshow(mask[yc:yc+h, x:x+w], alpha=0.6)
    plt.title('Cell Image + Mask')
    plt.axis('off')
    plt.show()


"""
Function to return the bounding box of the centroid in the mask
@input mask is a 2D-np array
@output rmin, rmax, cmin, cmax which are the (ymin, ymax, xmin, xmax) coordinates of the box
"""
def get_bounding_box(mask, expand=1.0):

    '''
    Gets exact coordinates of each cell in the mask
    ------------------------------------
    Argument:
    mask

    Returns:
    Rectangle coordinates
    '''

    assert expand >= 1.0

    n_rows, n_cols = mask.shape # get max height and width
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # max(..) because bounding box cannot exceed the dimension of the image
    # int(..) because coordinates cannot be floating points
    x_delta = int((cmax - cmin) * (expand - 1) / 2)
    y_delta = int((rmax - rmin) * (expand - 1) / 2)

    rmin = int(max(0, rmin - y_delta))
    cmin = int(max(0, cmin - x_delta))
    rmax = int(min(n_rows, rmax + y_delta))
    cmax = int(min(n_cols, cmax + x_delta))

    #assert rmin < rmax and cmin < cmax

    return rmin, rmax, cmin, cmax


def crop_and_resize_channel(channel, rmin, rmax, cmin, cmax, h_size=256, v_size=256):

    '''
    Helper function to Crop and resize each channel given the coordinates in the image
    ------------------------------------
    Argument:
    channel (R, G, B, Y)
    Coordinates of each cell in the image (rmin, rmax, cmin, cmax)

    Returns:
    New_image -- cropped and resized cell in the image
    '''

    cropped = channel[rmin:rmax, cmin:cmax]
    new_image = cv2.resize(cropped, (h_size, v_size))
    return new_image


def crop_and_resize_channels(channels, rmin, rmax, cmin, cmax, h_size=256, v_size=256, stack=True):
    '''
    Crop and resize all channels given the coordinates in the image
    ------------------------------------
    Argument:
    channels (RGBY) # channels can be either a list of arrays or a np.array
    Coordinates of each cell in the image (rmin, rmax, cmin, cmax)
    Default size is 256x256 pixel for standardization

    Returns:
    output -- np.array of each cropped and resized cell for all channels, if stack == True,    list otherwise
    '''

    num_channels = channels.shape[2]

    output = []
    for channel in range(num_channels):
        resized = crop_and_resize_channel(channels[:, :, channel], rmin, rmax, cmin, cmax, h_size, v_size)
        output.append(np.array(resized))
    if stack:
        return np.stack(output, axis=0) # stack along the 0th dim which is channel
    output = np.array(output)
    output = np.transpose(output, (1,2,0))
    return output


def apply_mask(mask, image):
    mask2 = np.stack((mask, mask, mask, mask), axis=2)
    out = np.multiply(mask2, image)
    return out


def plot_image_w_mask(mask, image):
    '''
    mask: mask with only one cell
    image: stacked RGBY image
    -------------------------
    output:
    merged image with mask
    '''
    shape = image.shape #array of (length, height, num_channels)
    mask2 = np.stack((mask, mask, mask, mask), axis=2)
    out = np.multiply(mask2, image)
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(out[:,:,:3])
    plt.title('image w mask')
    plt.axis('off')
    plt.show()

    return out


def plot_image_w_mask_3c(mask, image):
    '''
    mask: mask with only one cell
    image: stacked RGBY image
    -------------------------
    output:
    merged image with mask
    '''
    shape = image.shape #array of (length, height, num_channels)
    #mask = np.reshape(mask, (shape[0],shape[1],1))
    mask2 = np.stack((mask, mask, mask), axis=2)
    out = np.multiply(mask2, image)
    out.astype('float32')
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(out)
    plt.title('image w mask')
    plt.axis('off')
    plt.show()
    #print(out)

    return out


def plot_original_image(im):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(im[:,:,:3])
    plt.title('RGBY Image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join(ROOT, 'train.csv'))
    sample_sub = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'))
    # spliting label column
    train_df["Label"] = train_df["Label"].str.split("|")

    # class labels
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']


    # binarizing each label/class
    for label in tqdm(class_labels):
        train_df[label] = train_df['Label'].map(lambda result: 1 if label in result else 0)

    # rename column
    train_df.columns = ['ID', 'Label', 'Nucleoplasm', 'Nuclear membrane', 'Nucleoli', 'Nucleoli fibrillar center',
                        'Nuclear speckles', 'Nuclear bodies', 'Endoplasmic reticulum', 'Golgi apparatus', 'Intermediate filaments',
                        'Actin filaments', 'Microtubules', 'Mitotic spindle', 'Centrosome', 'Plasma membrane', 'Mitochondria',
                        'Aggresome', 'Cytosol', 'Vesicles and punctate cytosolic patterns', 'Negative']

    train = train_df.loc[train_df['Label'].apply(lambda x: len(x)==1)==True]
    class_counts = train.sum().drop(['ID', 'Label']).sort_values(ascending=False)
