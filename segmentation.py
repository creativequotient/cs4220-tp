import numpy as np
import cellsegmentator
import pandas a spd

for column in class_counts.keys():
    print(f"The class {column} has {train[column].sum()} samples")

NUC_MODEL = "../input/hpanucleimodel/dpn_unet_nuclei_v1.pth"
CELL_MODEL = "../input/hpacellmodel/dpn_unet_cell_3ch_v1.pth"

segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor=0.25,
    device="cpu",
    padding=False,
    multi_channel_model=True,
)

x_axis=[]
y_axis=[]

def preprocessing(train, start, end):
    for i in range(start, end+1):
        fileid = train.iloc[i].ID
        filelabel = train.iloc[i].Label
        #4 channels, stacked
        image = read_sample_image(fileid)
        im, r, b, y = read_sample_image_seg(fileid)
        mask = segmentCell(im, segmentator, seg_with="yellow")
        cell_num_array = np.unique(mask)
        for j in cell_num_array: #for each cell in the image
            trial_mask = deepcopy(mask)
            if j == 0:
                continue
            else:
                trial_mask[trial_mask > j] = 0
                trial_mask[trial_mask < j] = 0
                trial_mask[trial_mask == j] = 1
                #get bb box, crop and resize
                out_4 = apply_mask(trial_mask, image)
                rmin, rmax, cmin, cmax = get_bounding_box(trial_mask)
                new_image = crop_and_resize_channels(out_4, rmin, rmax, cmin, cmax, h_size=256, v_size=256, stack=False)
                x_axis.append(new_image)
                y_axis.append(filelabel)

preprocessing(train, 0, 10411)

df = pd.DataFrame()
df["labels"]=y_axis
df["images"]=x_axis
print(len(df), df.images[0].shape)
np.savez("x_images_arrays", x_axis)
np.savez("y_labels", y_axis)
