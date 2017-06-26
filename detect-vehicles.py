import glob
import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage.measurements as msrmnts
from PIL import Image
from moviepy.editor import VideoFileClip
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features_single(image, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=False, hist_feat=False, hog_feat=True):
    file_features = []
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)

    return np.concatenate(file_features)


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=False, hist_feat=False, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            print("spacial features length", len(spatial_features))
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            print("hist features length", len(hist_features))
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                print("hog features length", len(hog_features))

            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def extract_features_vis(imgs, cspace='RGB', orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):
    """Extract only hog freatures for visualization

    """
    # Create a list to append feature vectors to
    visualizations = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # vis=True so that we get the image
        _, visualization = get_hog_features(feature_image[:,:,hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        # Append the new visualization to the list
        visualizations.append(visualization)
    # Return list of visualizations
    return visualizations


def plot_feature_vectors(img_file, raw, normalized, file):
    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(img_file))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(raw)
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(normalized)
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.imsave(file)


def save_windows_on_image_to_file(windows, img, file):
    cp = np.copy(img)
    for window in windows:
        cv2.rectangle(cp, tuple(window[0]), tuple(window[1]), (0, 1, 0), thickness=3)
    plt.imsave(file, cp)


def draw_regions_of_interest(wsoi):
    f = plt.figure(figsize=(6, 11))  # we want axis so use figure
    for i, woi in enumerate(wsoi):
        img = mpimg.imread(woi['viz_file'])
        cv2.rectangle(img, (woi['x_start'], woi['y_start']), (woi['x_end'], woi['y_end']), woi['color'], thickness=3)
        plt.subplot(411 + i)
        plt.imshow(img)
        plt.title(woi['name'])
    f.tight_layout()
    f.savefig('output_images/rois.jpg')


def draw_windows_grid_for_regions_of_interest(wsoi):
    rsoi = get_regions_of_interest_viz(wsoi)
    f = plt.figure(figsize=(14, 11))  # we want axis so use figure
    for i, woi in enumerate(wsoi):
        roi = rsoi[i]
        w_w = woi['window'][0]
        w_h = woi['window'][1]
        steps_y, steps_x = get_window_steps(img_height=roi.shape[0], window_height=w_h,
                                            img_width=roi.shape[1], window_width=w_w,
                                            pixels_per_step=woi['pixels_per_step'])
        for step_x in steps_x:
            for step_y in steps_y:
                thickness = 1
                if step_x == 0 and step_y == 0:
                    thickness = 4
                cv2.rectangle(roi, (step_x, step_y), (step_x + w_w, step_y + w_h), (0, 1, 0), thickness=thickness)
        plt.subplot(411 + i)
        plt.imshow(roi)
        plt.title(woi['name'])
    f.tight_layout()
    f.savefig('output_images/windows_grid.jpg')


def get_regions_of_interest(img, wsoi):
    img_rsoi = []  # Regions of interest for all imgs
    for i, woi in enumerate(wsoi):
        img_roi_unscaled = img[woi['y_start']: woi['y_end'], woi['x_start']:woi['x_end']]
        img_rsoi.append(cv2.resize(img_roi_unscaled, (0, 0), fx=woi['scale'], fy=woi['scale']))
    return img_rsoi


def get_regions_of_interest_viz(wsoi):
    img_rsoi = []  # Regions of interest for all imgs
    for i, woi in enumerate(wsoi):
        img = cv2.cvtColor(cv2.imread(woi['viz_file']), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        img_roi_unscaled = img[woi['y_start']: woi['y_end'], woi['x_start']:woi['x_end']]
        img_rsoi.append(cv2.resize(img_roi_unscaled, (0, 0), fx=woi['scale'], fy=woi['scale']))
    return img_rsoi


# Compute window steps
def get_window_steps(img_height, window_height, img_width, window_width, pixels_per_step):
    cur_x = 0
    steps_x = []
    while cur_x + window_width <= img_width:
        steps_x.append(cur_x)
        cur_x += pixels_per_step
    cur_y = 0
    steps_y = []
    while cur_y + window_height <= img_height:
        steps_y.append(cur_y)
        cur_y += pixels_per_step
    return steps_y, steps_x


def get_hot_windows(woi, roi, svm, scaler):
    w_h = woi['window'][0]
    w_w = woi['window'][1]
    hot_windows = []
    steps_y, steps_x = get_window_steps(img_height=roi.shape[0], window_height=w_h,
                                        img_width=roi.shape[1], window_width=w_w,
                                        pixels_per_step=woi['pixels_per_step'])
    for step_x in steps_x:
        for step_y in steps_y:
            img_to_test = roi[step_y:step_y+w_h, step_x:step_x+w_w]
            features = extract_features_single(img_to_test, color_space=colorspace, orient=orient,
                                               pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                               hist_feat=True, spatial_feat=True, hog_channel=hog_channel)
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            prediction = svm.predict(test_features)
            if prediction == 1:
                hot_windows.append(((step_x, step_y), (step_x+w_w, step_y+w_h)))
    return np.asarray(hot_windows)


def get_all_hot_windows(wsoi, rsoi, img, svm, scalar, viz=False):
    hot_windows = []
    for i, woi in enumerate(wsoi):
        roi = rsoi[i]
        windows = get_hot_windows(woi, roi, svm, scalar)

        if viz:
            for window in windows:
                cv2.rectangle(roi, tuple(window[0]), tuple(window[1]), (0,1,0), thickness=3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(roi, "Region of Interest: {}".format(woi['name']), (20, 20), font, 0.5,
                        (1, 1, 1), 1)
            plt.imsave("output_images/windows_found_{}.jpg".format(woi['name']), roi)

        # Transform the results back to the coordinates of the original image
        for j, window in enumerate(windows):
            offset = np.asarray([woi['x_start'], woi['y_start']])
            windows[j] = (window / woi['scale']) + offset

        hot_windows.extend(windows)
    return hot_windows


def add_heat(black, windows):
    # Iterate through list of windows and add heat to each pixel in the window
    for window in windows:
        black[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    return black


def apply_heat_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap


def centeroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return np.asarray([sum_x/length, sum_y/length])


def draw_bboxes(img, bboxes_requested, centroids_history, prev_bboxes):
    """ Draw bounding boxes on img. Use distance between bboxes centroids and centroids from past 5 frames to reject
    bboxes that do not get 3 out 5 votes (from the previous 5 frames)
    """

    # centroids of bboxes requested to be drawn
    centroids = np.asarray([centeroid(np.asarray(bbox)) for bbox in bboxes_requested])

    # scores of the bboxes that are requested to be drawn, draw only if score > 3
    bboxes_scores = np.zeros(len(bboxes_requested))

    # list of bboxes that are drawn
    bboxes_drawn = []

    # look back 5 frames
    for centroids_frame in centroids_history:
        for i, bbox in enumerate(bboxes_requested):  # iterate over each bbox that is requested to be drawn
            bbox = np.asarray(bbox)
            bbox_centeroid = centeroid(bbox)
            if len(centroids_frame) > 0:     # only if we have data in the centroids frame should we compute the scores
                distances_to_centroids = np.sqrt(np.sum(np.square(centroids_frame - bbox_centeroid), axis=1))
                if np.min(distances_to_centroids) < 30:  # max 30 pixels away from previous centroid
                    bboxes_scores[i] += 1

    cp = np.copy(img)
    for i, bbox in enumerate(bboxes_requested):
        bbox = np.asarray(bbox)
        if len(centroids_history) < 5:  # no history so draw the boxes
            prev_bbox = find_previous_bbox(prev_bboxes, bbox)
            if prev_bbox is not None:
                bbox = low_pass_filter(np.asarray(bbox), np.asarray(prev_bbox), alpha=0.60)
            cv2.rectangle(cp, tuple(bbox[0]), tuple(bbox[1]), (0, 1, 0), thickness=3)
            bboxes_drawn.append(bbox)
        else:
            if bboxes_scores[i] >= 3:   # only accept bbox if 3+ out of 5 votes from previous frames
                prev_bbox = find_previous_bbox(prev_bboxes, bbox)
                if prev_bbox is not None:
                    bbox = low_pass_filter(np.asarray(bbox), np.asarray(prev_bbox), alpha=0.60)
                cv2.rectangle(cp, tuple(bbox[0]), tuple(bbox[1]), (0, 1, 0), thickness=3)
                bboxes_drawn.append(bbox)
    centroids_history.append(centroids)
    prev_bboxes[:] = bboxes_drawn
    return cp


def low_pass_filter(bbox, prev_bbox, alpha=0.6):
    return (bbox*alpha + (1-alpha)*prev_bbox).astype(np.int)


def find_previous_bbox(prev_bboxes, bbox):
    """Check if the bbox has a similar bbox in the prev_bboxes list. If so return that
    """
    for p_bbox in prev_bboxes:
        print(p_bbox)
        distance_1 = math.hypot(p_bbox[0][0]-bbox[0][0], p_bbox[0][1]-bbox[0][1])
        distance_2 = math.hypot(p_bbox[1][0]-bbox[1][0], p_bbox[1][1]-bbox[1][1])
        if distance_1 < 30 and distance_2 < 30:
            return p_bbox
    return None


def draw_car_annotations(img, windows, centroids_history=[], prev_bboxes=[], viz=False, file=''):
    # Setup the heatmap for the windows
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
    heatmap = add_heat(black=heatmap, windows=windows)
    heatmap = apply_heat_threshold(heatmap=heatmap, threshold=2)

    # Keep a copy of the centroids and bounding boxes
    if len(centroids_history) > 5:
        centroids_history.pop(0)

    # Annotate the cars in the image
    labels = msrmnts.label(heatmap)
    bboxes = []
    for car_number in range(1, labels[1]+1):  # Iterate through detected cars, labels[1] carries the number of elems
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    img_annotated = draw_bboxes(img, bboxes, centroids_history, prev_bboxes)

    # Visualize
    if viz:
        cp = np.copy(img)
        for window in windows:
            cv2.rectangle(cp, tuple(window[0]), tuple(window[1]), (0, 1, 0), thickness=3)
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.imshow(cp)
        plt.title('Car Positions')
        plt.subplot(132)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        plt.subplot(133)
        plt.imshow(img_annotated)
        plt.title('Cars Detected')
        fig.tight_layout()
        fig.savefig(file)

    # Return the image
    return img_annotated


def annotate_cars(image):
    image = image.astype(np.float32) / 255  # Training data was png, this is jpeg
    image_rsoi = get_regions_of_interest(image, wsoi)
    windows = get_all_hot_windows(wsoi, image_rsoi, image, svc, scaler)
    # using globals here, not the best way
    image_annotated = draw_car_annotations(img=image, windows=windows,
                                           centroids_history=centroids_history, prev_bboxes=prev_bboxes)
    image_annotated = (image_annotated * 255).astype(np.int8)
    return image_annotated

output_dir = '../output_images/'
viz = True
classify = False
colorspace = 'LUV'
hog_channel = 0  # 0,1,2
orient = 9
pix_per_cell = 8
cell_per_block = 2


# Load the vehicle and non-vehicle from the data directory
cars = np.asarray(glob.glob('data/vehicles/**/*.png', recursive=True))
notcars = np.asarray(glob.glob('data/non-vehicles/**/*.png', recursive=True))
print("Number of car images:", len(cars))
print("Number of not-car images:", len(notcars))
im_c = Image.open(cars[0])
im_nc = Image.open(cars[0])
print("Size (wxh) of a car image:", im_c.size)
print("Size (wxh) of a not-car image:", im_nc.size)

if viz:
    num_to_viz = 5
    idx_c = [np.random.randint(0, len(cars)) for i in range(0, num_to_viz)]
    idx_nc = [np.random.randint(0, len(notcars)) for i in range(0, num_to_viz)]
    f, axes = plt.subplots(num_to_viz, 2, figsize=(4.5, 11))
    f.tight_layout()
    for i in range(0, num_to_viz):
        img_c = cv2.cvtColor(cv2.imread(cars[idx_c[i]]), cv2.COLOR_BGR2RGB)
        img_nc = cv2.cvtColor(cv2.imread(notcars[idx_nc[i]]), cv2.COLOR_BGR2RGB)
        axes[i][0].set_title('Car Image', fontsize=8)
        axes[i][0].imshow(img_c)
        axes[i][0].axis('off')
        axes[i][1].set_title('Not-Car Image', fontsize=8)
        axes[i][1].imshow(img_nc)
        axes[i][1].axis('off')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    plt.subplots_adjust(hspace=.1, wspace=.05)
    f.savefig('output_images/sample.jpg')


# Apply HOG for visualization
if viz:
    num_to_viz = 3
    f, axes = plt.subplots(2, 2*num_to_viz+2, figsize=(17, 5))  # 2 images add manually
    f.tight_layout()
    idx_c = np.random.randint(0, len(cars), size=num_to_viz)
    idx_nc = np.random.randint(0, len(notcars), size=num_to_viz)
    imgs = np.concatenate([['test_images/car1.jpg', 'test_images/car2.jpg'], cars[idx_c], notcars[idx_nc]])
    imgs_h0 = extract_features_vis(imgs, cspace='LUV', orient=9,
                                   pix_per_cell=8, cell_per_block=2, hog_channel=0)
    for i, img in enumerate(imgs):
        img_ = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        axes[0][i].set_title('Source Image', fontsize=8)
        axes[0][i].imshow(img_)
        axes[0][i].axis('off')
        axes[1][i].set_title('HOG LUV Channel 0', fontsize=8)
        axes[1][i].imshow(imgs_h0[i])
        axes[1][i].axis('off')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    plt.subplots_adjust(hspace=.1, wspace=.05)
    f.savefig('output_images/hog_features.jpg')

# Define the feature vector
if classify:
    car_features = extract_features(cars[0:1], color_space=colorspace, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hist_feat=True, spatial_feat=True,  hog_channel=hog_channel)
    notcar_features = extract_features(notcars[0:0], color_space=colorspace, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hist_feat=True, spatial_feat=True, hog_channel=hog_channel)
    #X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X = np.vstack(car_features).astype(np.float64)

    # Normalize the feature vectors
    scaler = StandardScaler().fit(X)
    scaled_X = scaler.transform(X)

    if viz:
        idx = np.random.randint(0, 1)
        plot_feature_vectors(cars[idx], X[idx], scaled_X[idx], 'output_images/normalized_feature_vector.jpg')
    exit(0)



    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC to train a classifier
    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    joblib.dump(svc, 'svc_classifier.pkl')
    joblib.dump(scaler, 'scalar_classifier.pkl')
else:
    svc = joblib.load('svc_classifier.pkl')
    scaler = joblib.load('scalar_classifier.pkl')
    pass


# Windows of interest
wsoi = [
{
    'name': 'far',
    'y_start': 380,
    'y_end':  480,
    'x_start': 200,
    'x_end':  1080,
    'scale': 1.1,
    'pixels_per_step': 16,
    'window': (64, 64),   # (wxh)
    'color': (0, 255, 0),
    'viz_file': 'test_images/test7.jpg'
},
{
    'name': 'midway-far',
    'y_start': 380,
    'y_end':  530,
    'x_start': 200,
    'x_end':  1080,
    'scale': 0.9,
    'pixels_per_step': 16,
    'window': (64, 64),   # (wxh)
    'color': (0, 255, 0),
    'viz_file': 'test_images/test7.jpg'
},
{
    'name': 'midway-near',
    'y_start': 380,
    'y_end':  550,
    'x_start': 0,
    'x_end':  1280,
    'scale': 0.7,
    'pixels_per_step': 16,
    'window': (64, 64),   # (wxh)
    'color': (0, 255, 0),
    'viz_file': 'test_images/test1.jpg'
},
{
    'name': 'near',
    'y_start': 380,
    'y_end':  650,
    'x_start': 0,
    'x_end':  1280,
    'scale': 0.5,
    'pixels_per_step': 16,
    'window': (64, 64),   # (wxh)
    'color': (0, 255, 0),
    'viz_file': 'test_images/test8.jpg'
}]

if False:
    draw_regions_of_interest(wsoi)
    draw_windows_grid_for_regions_of_interest(wsoi)


if False:
    img1 = cv2.cvtColor(cv2.imread('test_images/test1.jpg'), cv2.COLOR_BGR2RGB)
    img1 = img1.astype(np.float32) / 255  # Training data was png, this is jpeg
    img1_rsoi = get_regions_of_interest(img1, wsoi)
    windows = get_all_hot_windows(wsoi, img1_rsoi, img1, svc, scaler, viz=True)
    save_windows_on_image_to_file(windows, img1, "output_images/windows_annotated_test1.jpg")
    draw_car_annotations(img1, windows, viz=True, file='output_images/heatmap_test1.jpg')

    img2 = cv2.cvtColor(cv2.imread('test_images/test6.jpg'), cv2.COLOR_BGR2RGB)
    img2 = img2.astype(np.float32) / 255  # Training data was png, this is jpeg
    img2_rsoi = get_regions_of_interest(img2, wsoi)
    windows = get_all_hot_windows(wsoi, img2_rsoi, img2, svc, scaler, viz=True)
    save_windows_on_image_to_file(windows, img2, "output_images/windows_annotated_test6.jpg")

    img3 = cv2.cvtColor(cv2.imread('test_images/test4.jpg'), cv2.COLOR_BGR2RGB)
    img3 = img3.astype(np.float32) / 255  # Training data was png, this is jpeg
    img3_rsoi = get_regions_of_interest(img3, wsoi)
    windows = get_all_hot_windows(wsoi, img3_rsoi, img3, svc, scaler, viz=True)
    save_windows_on_image_to_file(windows, img3, "output_images/windows_annotated_test4.jpg")

    img3 = cv2.cvtColor(cv2.imread('test_images/test7.jpg'), cv2.COLOR_BGR2RGB)
    img3 = img3.astype(np.float32) / 255  # Training data was png, this is jpeg
    img3_rsoi = get_regions_of_interest(img3, wsoi)
    windows = get_all_hot_windows(wsoi, img3_rsoi, img3, svc, scaler, viz=True)
    save_windows_on_image_to_file(windows, img3, "output_images/windows_annotated_test7.jpg")

    img = cv2.cvtColor(cv2.imread('test_images/test8.jpg'), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255  # Training data was png, this is jpeg
    img_rsoi = get_regions_of_interest(img, wsoi)
    windows = get_all_hot_windows(wsoi, img_rsoi, img, svc, scaler, viz=True)
    save_windows_on_image_to_file(windows, img, "output_images/windows_annotated_test8.jpg")

    img5 = cv2.cvtColor(cv2.imread('test_images/test9.jpg'), cv2.COLOR_BGR2RGB)
    img5 = img5.astype(np.float32) / 255  # Training data was png, this is jpeg
    img5_rsoi = get_regions_of_interest(img5, wsoi)
    windows = get_all_hot_windows(wsoi, img5_rsoi, img5, svc, scaler, viz=True)
    save_windows_on_image_to_file(windows, img5, "output_images/windows_annotated_test9.jpg")


if True:
    centroids_history = []
    prev_bboxes = []
    in_vid = 'project_video.mp4'
    out_vid = 'processed_' + in_vid.split('/')[-1]
    clip = VideoFileClip(in_vid).subclip(39,49)
    annotated_clip = clip.fl_image(annotate_cars)
    annotated_clip.write_videofile(out_vid, audio=False)
