# The following commands are meant for execution on kaggle notebooks
# !rsync -a ../input/mmdetection-v280/mmdetection ../
# !pip install ../input/mmdetection-v280/src/mmdet-2.8.0/mmdet-2.8.0/
# !pip install ../input/mmdetection-v280/src/mmpycocotools-12.0.3/mmpycocotools-12.0.3/
# !pip install ../input/mmdetection-v280/src/addict-2.4.0-py3-none-any.whl
# !pip install ../input/mmdetection-v280/src/yapf-0.30.0-py2.py3-none-any.whl
# !pip install ../input/mmdetection-v280/src/mmcv_full-1.2.6-cp37-cp37m-manylinux1_x86_64.whl

import pickle
from itertools import groupby
from pycocotools import mask as mutils
from pycocotools import _mask as coco_mask
import matplotlib.pyplot as plt
import os
import base64
import typing as t
import zlib
import random
random.seed(0)

exp_name = "v4"
conf_name = "mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco"
model_name = 'mask_rcnn_resnest101_v5_ep9'
ROOT = '../input/hpa-single-cell-image-classification/'
train_or_test = 'test'
df = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'))

if len(df) == 559:
    debug = True
    df = df[:3]
else:
    debug = False

def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != np.bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str.decode()


def read_img(image_id, color, train_or_test='train', image_size=None):
    filename = f'{ROOT}/{train_or_test}/{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    if img.dtype == 'uint16':
        img = (img/256).astype('uint8')
    return img


def load_RGBY_image(image_id, train_or_test='train', image_size=None):
    red = read_img(image_id, "red", train_or_test, image_size)
    green = read_img(image_id, "green", train_or_test, image_size)
    blue = read_img(image_id, "blue", train_or_test, image_size)
    # using rgb only here
    #yellow = read_img(image_id, "yellow", train_or_test, image_size)
    stacked_images = np.transpose(np.array([red, green, blue]), (1,2,0))
    return stacked_images


def print_masked_img(image_id, mask):
    img = load_RGBY_image(image_id, train_or_test)

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

config = f'configs/hpa_{exp_name}/{conf_name}.py'
model_file = f'../input/hpa-models/{model_name}.pth'
result_pkl = f'../work/{model_name}.pkl'
additional_conf = '--cfg-options'
additional_conf += ' test_cfg.rcnn.score_thr=0.001'
cmd = f'python tools/test.py {config} {model_file} --out {result_pkl} {additional_conf}'

# The following command was meant for execution on kaggle notebooks
# !cd ../mmdetection; {cmd}

result = pickle.load(open('../mmdetection/'+result_pkl, 'rb'))
for ii in range(3):
    image_id = annos[ii]['filename'].replace('.jpg','').replace('.png','')
    #image_id = '68ad8444-bb99-11e8-b2b9-ac1f6b6435d0'
    for class_id in range(19):
        #print(ii,class_id,len(result[ii][0][class_id]), len(result[ii][1][class_id]))
        bbs = result[ii][0][class_id]
        sgs = result[ii][1][class_id]
        for bb, sg in zip(bbs,sgs):
            box = bb[:4]
            cnf = bb[4]
            h = sg['size'][0]
            w = sg['size'][0]
            if cnf > 0.3:
                #print(f'class_id:{class_id}, image_id:{image_id}, confidence:{cnf}')
                mask = mutils.decode(sg).astype(bool)
                plt.imshow(mask)
                plt.title('Mask')
                plt.axis('off')
                img = load_RGBY_image(image_id)
                plt.imshow(img)
                plt.title('img')
                plt.axis('off')
                plt.imshow(img)
                plt.imshow(mask, alpha=0.6)
                plt.title('Image + Mask')
                plt.axis('off')
                plt.show()
                #print_masked_img(image_id, mask)
