# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # HumanEva Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC To train and evaluate our ensembled models for human pose estimation, we decided to use [HumanEva-I](http://humaneva.is.tue.mpg.de/) (or simply HumanEva) dataset. It consists of three subjects performing a set of six pre-defined actions including `Walk`, `Jog`, `Throw/catch`, `Gesture`, `Box`, and `Combo`. The subjects movements were recorded using three synchronized cameras at 60 Hz, and ground-truth 3D motions were simultaneously captured using a commercial motion capture system. In total, there are 56 sequences and approximately 80000 frames. As a human pose model, a 15-joint skeleton was adopted, giving 15 keypoints. Our approach of distributed ensembles can be used for other 3D pose estimation datasets as long as the 2D/3D locations are provided. 

# COMMAND ----------

# MAGIC %md
# MAGIC [![humaneva](https://www.researchgate.net/profile/Meng_Ding2/publication/261387338/figure/fig3/AS:649320483278848@1531821475103/Illustration-of-experimental-results-on-Subject-I-in-HumanEva-dataset-The-estimated_W640.jpg)](https://www.youtube.com/watch?v=W2xt0RKItVk)

# COMMAND ----------

# MAGIC %md
# MAGIC The data and future model checkpoints will be loaded under `dbfs:/VideoPose3D`. 

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/VideoPose3D/saved_models/humaneva/checkpoints/semi_supervised

# COMMAND ----------

# MAGIC %md
# MAGIC We start by loading the data that has been pre-processed by the authors of [1] and [2]. Further down we show how we transform it into the `.csv` format, which is well supported by Apache Spark.
# MAGIC 
# MAGIC <!-- The train and test data are loaded as DataFrames. The columns consists of frame index, subject index, action name, camera index, and 2D and 3D positions of each joint (keypoint). For brevity, the subject indices, action names and camera indices are grouped together to a single column. — this is discussed in the next notebook. -->

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2D data and ground-truth 3D poses

# COMMAND ----------

# MAGIC %md
# MAGIC ### Keypoints

# COMMAND ----------

# MAGIC %md
# MAGIC First we load the two `.npz` files with the 2D and 3D keypoints of the HumanEva dataset, respectively.

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install pip --upgrade --quiet
# MAGIC pip install gdown --quiet
# MAGIC cd /dbfs/VideoPose3D/humaneva
# MAGIC gdown 1EngBymOjXWPntjfNVaGZhBX7sNCNg9pu # data_2d_humaneva15_gt.npz
# MAGIC gdown 1ErTRudqF8ugAwopL3ieral0YMEtE28Dd # data_3d_humaneva15.npz

# COMMAND ----------

# MAGIC %md
# MAGIC * `data_2d_humaneva15_gt.npz` contains `pos2d` with 2D keypoint locations of the joints of moving humans in various video sequences. The format is as following:
# MAGIC   * it is a dictionary with keys corresponding to different subjects: `S1`, `S2`, and `S3`
# MAGIC   * since it was pre-split into train-validation data by the dataset authors, the keys we see are `Train/S1`... `Valiation/S1`..., however we ignore that split
# MAGIC   * each subject contains another dictionary with keys corresponding to different actions: `Jog`, `Box`, `Walking`, `Gestures`, `ThrowCatch`.
# MAGIC   * again, since it was pre-split, instead of the full videos we get the chunks of videos, `Jog chunk0`...
# MAGIC   * for each video, we have three views (`camera` can be `0`, `1`, or `2`), since three cameras were looking at the moving subjects during data collection
# MAGIC 
# MAGIC * `data_3d_humaneva15.npz` contains `pos3d` which has the same structure as `pos2d`, but instead of the 2D keypoint locations, it contains the ground-truth 3D keypoint locations, and also it doesn't have three different views.
# MAGIC 
# MAGIC We transform the `.npz` files into the `.csv` files that will be used further when working with RDDs. We split the data into train and test subsets for convenience and make sure that both contain a reasonable portion of data for each subject and each action.

# COMMAND ----------

import numpy as np
import pandas as pd
ROOTDIR = '/dbfs/VideoPose3D'

path2d = f'{ROOTDIR}/humaneva/data_2d_humaneva15_gt.npz'
path3d = f'{ROOTDIR}/humaneva/data_3d_humaneva15.npz'
pos2d = np.load(path2d, allow_pickle=True)['positions_2d'].item()
pos3d = np.load(path3d, allow_pickle=True)['positions_3d'].item()
pos_data = []
pos_data_train = []
pos_data_test = []
assert(pos2d.keys() == pos3d.keys())
for subject in pos2d.keys():
    print(f'{subject}: {sum([pos2d[subject][action][0].shape[0] for action in pos2d[subject].keys()])} frames in total')
    assert(pos2d[subject].keys() == pos3d[subject].keys())
    print(list(pos2d[subject].keys()))
    
    # Train-Test split
    actions_for_test = [[a for a in pos2d[subject].keys() if action_name in a] for action_name in ['Jog', 'Box', 'Walking', 'Gestures', 'ThrowCatch']]
    actions_for_test = [a[1] for a in actions_for_test if len(a)>1]

    # Add to full data
    for action in pos2d[subject].keys():
        for camera in [0,1,2]:
            n_frames = pos2d[subject][action][camera].shape[0]
            assert(n_frames==pos3d[subject][action].shape[0])
            frames = np.hstack([
                pos2d[subject][action][camera].reshape(n_frames,-1),
                pos3d[subject][action].reshape(n_frames,-1)])
            row = [[subject, action, camera, *frame] for frame in frames]
            pos_data.extend(row)

    # Add to train data
    for action in set(pos2d[subject].keys()) - set(actions_for_test):
        for camera in [0,1,2]:
            n_frames = pos2d[subject][action][camera].shape[0]
            assert(n_frames==pos3d[subject][action].shape[0])
            frames = np.hstack([
                pos2d[subject][action][camera].reshape(n_frames,-1),
                pos3d[subject][action].reshape(n_frames,-1)])
            row = [[subject, action, camera, *frame] for frame in frames]
            pos_data_train.extend(row)

    # Add to test data
    for action in actions_for_test:
        for camera in [0,1,2]:
            n_frames = pos2d[subject][action][camera].shape[0]
            assert(n_frames==pos3d[subject][action].shape[0])
            frames = np.hstack([
                pos2d[subject][action][camera].reshape(n_frames,-1),
                pos3d[subject][action].reshape(n_frames,-1)])
            row = [[subject, action, camera, *frame] for frame in frames]
            pos_data_test.extend(row)

print('Creating full dataframe...')
pos_df = pd.DataFrame(pos_data, columns=['Subject','Action','Camera'] + (','.join([f'x{i+1},y{i+1}' for i in range(15)])).split(',') + (','.join([f'X{i+1},Y{i+1},Z{i+1}' for i in range(15)])).split(','))

print('Creating train dataframe...')
pos_df_train = pd.DataFrame(pos_data_train, columns=['Subject','Action','Camera'] + (','.join([f'x{i+1},y{i+1}' for i in range(15)])).split(',') + (','.join([f'X{i+1},Y{i+1},Z{i+1}' for i in range(15)])).split(','))

print('Creating test dataframe...')
pos_df_test = pd.DataFrame(pos_data_test, columns=['Subject','Action','Camera'] + (','.join([f'x{i+1},y{i+1}' for i in range(15)])).split(',') + (','.join([f'X{i+1},Y{i+1},Z{i+1}' for i in range(15)])).split(','))

SAVE = False
if SAVE:
    pos_df.to_csv(f'{ROOTDIR}/humaneva/humaneva15.csv')
    pos_df_train.to_csv(f'{ROOTDIR}/humaneva/humaneva15_train.csv')
    pos_df_test.to_csv(f'{ROOTDIR}/humaneva/humaneva15_test.csv')
print('Done.')

# COMMAND ----------

# MAGIC %md
# MAGIC We also experimented with 2D keypoint detections produced by Mask-RCNN, which we load in the cell below.

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /dbfs/VideoPose3D/humaneva/
# MAGIC wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_humaneva15_detectron_pt_coco.npz

# COMMAND ----------

# MAGIC %md
# MAGIC ### Skeleton

# COMMAND ----------

# MAGIC %md
# MAGIC We pre-save the skeleton data for HumanEva dataset.

# COMMAND ----------

# MAGIC %python
# MAGIC humaneva_skeleton = {
# MAGIC     'parents': [-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
# MAGIC     'joints_left': [2, 3, 4, 8, 9, 10],
# MAGIC     'joints_right': [5, 6, 7, 11, 12, 13],
# MAGIC }
# MAGIC np.savez(f'{ROOTDIR}/humaneva/humaneva_skeleton.npz', data=humaneva_skeleton)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC Let's plot the first frames of the video sequences in our dataset to understand, what kind of input is expected by the neural network (see more details on that in the next noteook).

# COMMAND ----------

from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

desc_list = []
pos2d_list = []
for row in pos_df.iterrows():
    if row[1].values[2]==0:
        desc_list.append(': '.join(row[1].values[:2]) + f' (Cam {row[1].values[2]})')
        pos2d_list.append(np.array(row[1].values[3:3+15*2]).reshape(15,2))
pos2d_list = np.array(pos2d_list)

figure, ax = plt.subplots(figsize=(10,10))
ax.axis('equal')
xmin, xmax = pos2d_list[:,:,0].min(), pos2d_list[:,:,0].max()
ymin, ymax = pos2d_list[:,:,1].min(), pos2d_list[:,:,1].max()

def animation_function(i):
    ax.clear()
    # Setting title as subject + action + camera
    ax.set_title(desc_list[i])
    # Setting limits for x and y axis
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    # Plotting the 2D keypoints
    x = pos2d_list[2*i,:,0]
    y = pos2d_list[2*i,:,1]
    plt.scatter(x, y)
    # Plotting the 2D skeleton
    for indices in [np.array([-1, 1, 0]),
                    np.array([1, 2, 3, 4]),
                    np.array([1, 5, 6, 7]),
                    np.array([0, 8, 9, 10]),
                    np.array([0, 11, 12, 13])]:
        plt.plot(x[indices], y[indices], 'b-')

animation = FuncAnimation(figure, animation_function, frames=500)
HTML(animation.to_jshtml())