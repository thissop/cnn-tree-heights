#NB_EPOCHS, MAX_TRAIN_STEPS
import os
from cnnheights.training import train_cnn
from cnnheights.prediction import predict
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ndvi_images = []
pan_images = [] 
annotations = [] 
boundaries = []

# [1] 
# python /ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/src/monthly/jan2023/library-testing/first_training_test.py > /ar1/PROJ/fjuhsd/personal/thaddaeus/other/cnn-heights/output/log.txt &

computer = 'wh1' # input('m2, wh1, or wsl: ')

if computer == 'm2': 
    data_dir = '/Users/yaroslav/Documents/Work/NASA/data/old/ready-for-cnn/cnn-input/'
    logging_dir = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/library-testing/cnn-training-output'

elif computer == 'wh1': 
    data_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/cnn-input/'
    logging_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/src/monthly/feb2023/misc/confusion-matrix/boilerplate-output'

elif computer == 'wsl': 
    data_dir = ''

else:
    raise Exception('Choose correct computer to work on!')

for file in np.sort(os.listdir(data_dir)):
    full_path = data_dir+file
    if '.png' in file: 
        if 'annotation' in file: 
            annotations.append(full_path) 
 
        elif 'boundary' in file: 
            boundaries.append(full_path) 

        elif 'ndvi' in file: 
            ndvi_images.append(full_path) 

        elif 'extracted_pan' in file: 
            pan_images.append(full_path) 

def train_predict():
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import seaborn as sns
    import matplotlib.pyplot as plt 

    model, hist = train_cnn(ndvi_images, pan_images, annotations, boundaries, logging_dir=logging_dir, epochs=1, training_steps=7, use_multiprocessing=True)

    mask = predict(model, ndvi_images[0], pan_images[0], output_dir='/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/src/monthly/feb2023/misc/confusion-matrix/boilerplate-output', crs='EPSG:32628')

    predictions = mask.flatten()
    predictions[predictions>1] = 1
    predictions[predictions<1] = 0
    predictions = predictions.astype(int)

    image = Image.open(annotations[0])
    annotation_data = np.asarray(image).flatten()
    annotation_data[annotation_data>1] = 1
    annotation_data[annotation_data<1] = 0
    annotation_data = annotation_data.astype(int)

    fig, ax = plt.subplots()
    cm_labels = ['0', '1']

    cm = confusion_matrix(annotation_data, predictions, normalize='pred')
    sns.heatmap(cm, annot=True, linewidths=.5, ax=ax, center=0.0, yticklabels=cm_labels, xticklabels=cm_labels, annot_kws={'fontsize':'xx-large'})
    
    ax.set(xlabel='True Class', ylabel='Predicted Class')#, xticks=[0,1], yticks=[0,1])
    #ax.axis('off')
    #ax.tick_params(top=False, bottom=False, left=False, right=False)

    fig.tight_layout()

    plt.savefig('src/monthly/feb2023/misc/confusion-matrix/confusion-matrix.pdf')

    tn, fp, fn, tp = cm.ravel()
    
train_predict()

def testing(): 

    import geopandas as gpd
    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots()

    gdf = gpd.read_file('src/monthly/feb2023/misc/confusion-matrix/boilerplate-output/predicted_polygons.shp')
    
    gdf = gdf[gdf.geom_type != 'MultiPolygon']

    print(gdf)
    gdf.plot(ax=ax)

    plt.savefig('src/monthly/feb2023/misc/confusion-matrix/pred.pdf')

    plt.close()

    fig, ax = plt.subplots()

    image = Image.open(annotations[0])
    # convert image to numpy array
    annotation_data = np.asarray(image).flatten()
    annotation_data[annotation_data>1] = 1
    annotation_data[annotation_data<1] = 0

    #print(annotation_data.shape) # (279, 362)

    plt.imshow(np.asarray(Image.open(annotations[0])))
    #ax.hist(annotation_data)
    plt.savefig('src/monthly/feb2023/misc/confusion-matrix/annotation.pdf')

    print(annotation_data)

#testing()