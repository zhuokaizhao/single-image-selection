import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# image hover function for 2D scatter plot
def hover_2d(event):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w, h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy =(pca_results[ind, 0], pca_results[ind, 1])
        # set the image corresponding to that point
        im.set_data(all_original_images[ind])
        print(f'Current seletected image id: {ind}')
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)

    fig.canvas.draw_idle()


# image hover function for 3D scatter plot
def hover_3d(event):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w, h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xyz =(pca_results[ind, 0], pca_results[ind, 1], pca_results[ind, 2])
        # set the image corresponding to that point
        im.set_data(all_original_images[ind])
        print(f'Current seletected image id: {ind}')
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)

    fig.canvas.draw_idle()

# a few parameters
data_type = 'train'
target_type = 'single-test'
PCA_component = 3
image_size = (128, 128)

# optionally input selection result
test_result_path = 'all_datasets_results_train.npy'
# input images dir
original_image_dir = f'original_images/from_{data_type}'

# load testing result
test_result = np.load(test_result_path)
print(f'\nLoaded test result with shape {test_result.shape}')
# get paths for all images
all_original_image_paths = os.listdir(original_image_dir)
# load all images
all_original_images = []
for original_image_path in all_original_image_paths:
    image = Image.open(os.path.join(original_image_dir, original_image_path))
    # resizes image in-place
    image.thumbnail((512, 512), Image.BILINEAR)
    all_original_images.append(np.array(image))

print(f'Loaded original images with shape {all_original_images[0].shape}')

# scale the features in your data before applying PCA
test_result = StandardScaler().fit_transform(test_result)
# run PCA
pca = PCA(n_components=PCA_component)
pca_results = pca.fit_transform(test_result)
print(f'PCA result array has shape {pca_results.shape}')

# visualize 2D projection
fig = plt.figure(figsize = (8, 8))

# create the annotations box
im = OffsetImage(all_original_images[0], zoom=1)
xybox = (50., 50.)
ab = AnnotationBbox(im, (0, 0), xybox=xybox, xycoords='data',
                    boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))

if PCA_component == 2:
    ax = fig.add_subplot(1,1,1)
    line = ax.scatter(pca_results[:, 0],
                        pca_results[:, 1],
                        c = 'blue',
                        s = 30)
    ax.set_title(f'PCA result of {len(test_result)} single-test candidates', fontsize = 15)
    ax.set_xlabel('PC 1', fontsize = 10)
    ax.set_ylabel('PC 2', fontsize = 10)
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)
    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover_2d)

elif PCA_component == 3:
    ax = fig.add_subplot(projection='3d')
    line = ax.scatter(pca_results[:, 0],
                        pca_results[:, 1],
                        pca_results[:, 2],
                        c = 'blue',
                        marker = 'o')
    ax.set_title(f'PCA result of {len(test_result)} single-test candidates', fontsize = 15)
    ax.set_xlabel('PC 1', fontsize = 10)
    ax.set_ylabel('PC 2', fontsize = 10)
    ax.set_zlabel('PC 3', fontsize = 10)
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)
    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover_3d)


ax.grid()
plt.show()