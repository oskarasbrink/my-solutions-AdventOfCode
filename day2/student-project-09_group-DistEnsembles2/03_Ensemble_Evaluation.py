# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluating Ensemble Performance

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functionality from the previous notebook

# COMMAND ----------

import json
import numpy as np
import torch
from random import sample, seed
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType
from pyspark.sql.functions import collect_list, size, udf
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import udf
from itertools import groupby
from pyspark.rdd import PipelinedRDD

from pathlib import Path
import os
import matplotlib.pyplot as plt

ROOTDIR = '/VideoPose3D'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data functionality

# COMMAND ----------

humaneva_train_path = f'{ROOTDIR}/humaneva/humaneva15_train.csv'
humaneva_test_path = f'{ROOTDIR}/humaneva/humaneva15_test.csv'

def load_data_from_csv(file_location):
    file_type = "csv"
    infer_schema = "true"
    first_row_is_header = False
    delimiter = ","
    
    schema = StructType() \
      .add("Idx",IntegerType(),True) \
      .add("Subject",StringType(),True) \
      .add("Action",StringType(),True) \
      .add("Camera",StringType(),True)
    for i in range(15):
        schema = schema.add(f"u{i}",DoubleType(),True).add(f"v{i}",DoubleType(),True)
    for i in range(15):
        schema = schema.add(f"X{i}",DoubleType(),True).add(f"Y{i}",DoubleType(),True).add(f"Z{i}",DoubleType(),True)
    
    # Load the data from file
    df = spark.read.csv(file_location, header=True, schema=schema, sep=',')
    return df

df_train = load_data_from_csv(humaneva_train_path).withColumn("Group", F.concat_ws(', ', "Subject", "Action", "Camera")).drop("Subject", "Action", "Camera")
df_test = load_data_from_csv(humaneva_test_path).withColumn("Group", F.concat_ws(', ', "Subject", "Action", "Camera")).drop("Subject", "Action", "Camera")

feature_names = []
target_names = []
n_keypoints = 15 
for i in range(n_keypoints):
    feature_names.append("u{}".format(i))
    feature_names.append("v{}".format(i))
    target_names.append("X{}".format(i))
    target_names.append("Y{}".format(i))
    target_names.append("Z{}".format(i))

feature_assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
target_assembler = VectorAssembler(inputCols=target_names, outputCol="targets")


def assemble_vectors(df):
    df = feature_assembler.transform(df)
    df = target_assembler.transform(df)
    df = df.drop(*feature_names).drop(*target_names)
    return df

df_train = assemble_vectors(df_train)
df_test = assemble_vectors(df_test)

receptive_field = 27

w = Window.orderBy("Idx").partitionBy(["Group"]).rowsBetween(Window.currentRow-receptive_field//2, Window.currentRow+receptive_field//2)

def create_receptive_fields(df):
    df = df.withColumn("feature_sequence", collect_list("features").over(w))
    df = df.withColumn("group_sequence", collect_list("Group").over(w))
    df = df.filter(size(df.group_sequence) == receptive_field)
    return df

df_train_receptive = create_receptive_fields(df_train).drop("features")
df_test_receptive = create_receptive_fields(df_test).drop("features")

seed(0)
chunks = df_train_receptive.select("Group").distinct().collect()
chunks = [x["Group"] for x in chunks]

num_chunks = len(chunks)
num_unlabeled = int(num_chunks*0.6)

unlabeled_chunks = sample(chunks, num_unlabeled)
labeled_chunks = [x for x in chunks if x not in unlabeled_chunks]

df_train_receptive_unlabeled = df_train_receptive.filter(df_train_receptive.Group.isin(unlabeled_chunks))
df_train_receptive_unlabeled = df_train_receptive_unlabeled.drop("targets")
df_train_receptive_labeled = df_train_receptive.filter(~df_train_receptive.Group.isin(unlabeled_chunks))


def toTensorLabeled(x):
    fs = x["feature_sequence"]
    target = x["targets"]
    
    feature_tensor = []
    for f in fs:
        feature_tensor.append(f)
    
    xx = torch.tensor(feature_tensor,dtype=torch.float)
    yy = torch.tensor(target,dtype=torch.float)
    
    return xx.view(27, 15, 2), yy.view(1, 15, 3)

def toTensorUnlabeled(x):
    fs = x["feature_sequence"]
    feature_tensor = []
    for f in fs:
        feature_tensor.append(f)
    
    xx = torch.tensor(feature_tensor, dtype=torch.float)
    
    return xx.view(27, 15, 2)

labeled_tensor_rdd = df_train_receptive_labeled.rdd.map(toTensorLabeled)
unlabeled_tensor_rdd = df_train_receptive_unlabeled.rdd.map(toTensorUnlabeled)
test_tensor_rdd = df_test_receptive.rdd.map(toTensorLabeled)
labeled_full_size = labeled_tensor_rdd.count()
unlabeled_full_size = unlabeled_tensor_rdd.count()
test_full_size = test_tensor_rdd.count()


class DataSet(torch.utils.data.Dataset):
    def __init__(self, pos2D, pos3D):
        self.pos2D = pos2D # self.pos2D: B x 27 x 15 * 2
        self.pos3D = pos3D # self.pos3D: B x 1 x 15 * 3

    def __len__(self):
        return self.pos2D.shape[0]

    def __getitem__(self, ind):
        pos2D = self.pos2D[ind] # pos2D: B x 27 x 15 * 2 -> 27 x 15 * 2
        pos3D = self.pos3D[ind] # pos2D: B x 1 x 15 * 3 -> 1 x 15 * 2
        return pos2D, pos3D

data_labeled = labeled_tensor_rdd.takeSample(False, labeled_full_size)
pos2D_labeled, pos3D_labeled = zip(*data_labeled)
pos2D_labeled, pos3D_labeled = torch.stack(pos2D_labeled), torch.stack(pos3D_labeled)

data_test = test_tensor_rdd.takeSample(False, test_full_size)
pos2D_test, pos3D_test = zip(*data_test)
pos2D_test, pos3D_test = torch.stack(pos2D_test), torch.stack(pos3D_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model functionality

# COMMAND ----------

# MAGIC %python
# MAGIC from torch import nn
# MAGIC 
# MAGIC class Args:
# MAGIC     num_joints = 15
# MAGIC     stride = 1    # temporal length of the prediction to use during training
# MAGIC     epochs = 10    # number of training epochs
# MAGIC     batch_size = 128     # batch size in terms of predicted frames
# MAGIC     dropout = 0.25    # dropout probability
# MAGIC     learning_rate = 0.001    # initial learning rate
# MAGIC     lr_decay = 0.996     # learning rate decay per epoch
# MAGIC     data_augmentation = True # disable train-time flipping
# MAGIC     test_time_augmentation = True # disable test-time flipping
# MAGIC     architecture = '3,3,3'    # filter widths separated by comma
# MAGIC     channels = 1024    # number of channels in convolution layers
# MAGIC 
# MAGIC args = Args()
# MAGIC filter_widths = [int(x) for x in args.architecture.split(',')]
# MAGIC receptive_field = np.prod(filter_widths) # model_pos.receptive_field()
# MAGIC print('INFO: Receptive field: {} frames'.format(receptive_field))
# MAGIC pad = (receptive_field - 1) // 2 # Padding on each side
# MAGIC hyperparams = [args.num_joints, 2, args.num_joints, filter_widths, args.dropout, args.channels]
# MAGIC 
# MAGIC class TemporalModelBase(nn.Module):
# MAGIC     def __init__(self, num_joints_in, in_features, num_joints_out,
# MAGIC                  filter_widths, dropout, channels):
# MAGIC         super().__init__()
# MAGIC         # Validate input
# MAGIC         for fw in filter_widths:
# MAGIC             assert fw % 2 != 0, 'Only odd filter widths are supported'
# MAGIC         self.num_joints_in = num_joints_in
# MAGIC         self.in_features = in_features
# MAGIC         self.num_joints_out = num_joints_out
# MAGIC         self.filter_widths = filter_widths
# MAGIC         self.drop = nn.Dropout(dropout)
# MAGIC         self.relu = nn.ReLU(inplace=True)
# MAGIC         self.pad = [ filter_widths[0] // 2 ]
# MAGIC         self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
# MAGIC         self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)
# MAGIC         
# MAGIC 
# MAGIC     def set_bn_momentum(self, momentum):
# MAGIC         self.expand_bn.momentum = momentum
# MAGIC         for bn in self.layers_bn:
# MAGIC             bn.momentum = momentum
# MAGIC         
# MAGIC     def forward(self, pos2D):
# MAGIC         assert len(pos2D.shape) == 4 # pos2D: B x 27 x 15 x 2
# MAGIC         assert pos2D.shape[-2] == self.num_joints_in # 15
# MAGIC         assert pos2D.shape[-1] == self.in_features   # 2     
# MAGIC         sz = pos2D.shape[:3] # B x 27 x 15
# MAGIC         pos2D = pos2D.view(pos2D.shape[0], pos2D.shape[1], -1) # B x 27 x 15 * 2
# MAGIC         pos2D = pos2D.permute(0, 2, 1) # B x 15 * 2 x 27
# MAGIC         pos3D = self._forward_blocks(pos2D)
# MAGIC         pos3D = pos3D.permute(0, 2, 1)
# MAGIC         pos3D = pos3D.view(sz[0], -1, self.num_joints_out, 3)
# MAGIC         return pos3D
# MAGIC 
# MAGIC class TemporalModel(TemporalModelBase):
# MAGIC     def __init__(self, num_joints_in, in_features, num_joints_out,
# MAGIC                  filter_widths, dropout=0.25, channels=1024):
# MAGIC         """
# MAGIC         Reference 3D pose estimation model with temporal convolutions.Initialize this model.
# MAGIC         
# MAGIC         Arg:
# MAGIC             num_joints_in -- number of input joints (i.e. 15 for HumanEva-I)
# MAGIC             in_features -- number of input features for each joint (typically 2 for 2D input)
# MAGIC             num_joints_out -- number of output joints (can be different than input)
# MAGIC             filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
# MAGIC             dropout -- dropout probability
# MAGIC             channels -- number of convolution channels
# MAGIC         """
# MAGIC         super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, dropout, channels)
# MAGIC         self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)
# MAGIC         layers_conv = []
# MAGIC         layers_bn = []
# MAGIC         next_dilation = filter_widths[0] # 3
# MAGIC         for i in range(1, len(filter_widths)):
# MAGIC             self.pad.append((filter_widths[i] - 1)*next_dilation // 2) # [1, 3, 9]
# MAGIC             layers_conv.append(nn.Conv1d(channels, channels,
# MAGIC                                          filter_widths[i],
# MAGIC                                          dilation=next_dilation,
# MAGIC                                          bias=False))
# MAGIC             layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
# MAGIC             layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
# MAGIC             layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
# MAGIC             next_dilation *= filter_widths[i] # 3, 9, 27
# MAGIC         self.layers_conv = nn.ModuleList(layers_conv)
# MAGIC         self.layers_bn = nn.ModuleList(layers_bn)
# MAGIC         
# MAGIC     def _forward_blocks(self, pos2D):
# MAGIC         # pos2D: B x 15 * 2 x 27
# MAGIC         x = self.drop(self.relu(self.expand_bn(self.expand_conv(pos2D)))) # B x 1024 x 25
# MAGIC         for i in range(len(self.pad) - 1):
# MAGIC             pad = self.pad[i+1] # 3, 9
# MAGIC             res = x[:, :, pad : x.shape[2] - pad] # B x 1024 x 19, B x 1024 x 1
# MAGIC             x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x)))) # B x 1024 x 19, B x 1024 x 1
# MAGIC             x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
# MAGIC         pos3D = self.shrink(x) # B x 15*3 x 1
# MAGIC         return pos3D
# MAGIC     
# MAGIC     @staticmethod
# MAGIC     def from_state_dict(params, hyperparams):
# MAGIC         net = TemporalModel(*hyperparams)
# MAGIC         net.load_state_dict(params)
# MAGIC         return net
# MAGIC 
# MAGIC def mpjpe(predicted, target):
# MAGIC     """
# MAGIC     Mean per-joint position error (i.e. mean Euclidean distance),
# MAGIC     often referred to as "Protocol #1" in many papers.
# MAGIC     """
# MAGIC     assert predicted.shape == target.shape
# MAGIC     return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation protocol

# COMMAND ----------

def evaluate(models, pos2D, pos3D, args):
    for model in models:
        model.eval()
    with torch.no_grad():
        inputs_2d, inputs_3d = pos2D, pos3D
        predicted_3d_pos = [model(inputs_2d) for model in models]
        predicted_3d_pos = sum(predicted_3d_pos) / len(predicted_3d_pos)
        loss_3d = mpjpe(predicted_3d_pos, inputs_3d).item()
        N = 1
        print(f'\t{loss_3d}')
    return loss_3d/N
  
def get_test_predictions(models, pos2D, args):
    for model in models:
        model.eval()
    with torch.no_grad():
        inputs_2d = pos2D
        predicted_3d_pos = [model(inputs_2d) for model in models]
    return predicted_3d_pos

def eval_ensemble_size(n_members, iters):
    test_scores = []
    for i in iters:
        models = []
        for j in range(n_members):
            params_path = f'/dbfs/VideoPose3D/saved_models/humaneva/checkpoints/supervised/{n_members}_members_ensemble{j}_iter{i}.ckpt'
            params = torch.load(params_path, map_location=torch.device("cpu"))
            models.append(TemporalModel.from_state_dict(params, hyperparams))
        test_score = evaluate(models, pos2D_test, pos3D_test, args)
        test_scores.append(test_score)
    return test_scores
 
def eval_ensemble_size_semi_supervised(n_members, iters):
    test_scores = []
    for i in iters:
        models = []
        for j in range(n_members):
            params_path = f'/dbfs/VideoPose3D/saved_models/humaneva/checkpoints/semi_supervised/{n_members}_members_ensemble{j}_iter{i}.ckpt'
            params = torch.load(params_path) #, map_location=torch.device("cpu")
            models.append(TemporalModel.from_state_dict(params, hyperparams))
        test_score = evaluate(models, pos2D_test, pos3D_test, args)
        test_scores.append(test_score)
    return test_scores    

# COMMAND ----------

# MAGIC %md
# MAGIC ####Extracting checkpoints of supervised models.

# COMMAND ----------

x = list(range(0,100,5))
x.append(99)

ensemble_sizes = [1, 2, 3, 5]
#ensemble_sizes = [5]
test_scores_dict = {}

for size in ensemble_sizes:
    test_scores_dict[size] = eval_ensemble_size(size, x)

# COMMAND ----------

# save results
with open("/dbfs/VideoPose3D/saved_results.txt", "w") as results_file:
    results_file.write(json.dumps(test_scores_dict))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test errors during training for different ensemble sizes
# MAGIC 
# MAGIC A training iteration here is when the models are trained on a subset of the full data for a number of epochs. Surprisingly, we don't see a clear reduction of test error when increasing the ensemble size. It does however seem that the test errors are slightly more stable with increasing ensemble size.

# COMMAND ----------

for size, test_scores in test_scores_dict.items():
    plt.plot(x, test_scores, label="Ensemble size: {}".format(size))

plt.ylabel("Test error")
plt.xlabel("Training iterations")    
plt.legend()
plt.show()

# COMMAND ----------

x = list(range(0,100,5))
x.append(99)

test_scores_dict_loaded = json.load(open("/dbfs/VideoPose3D/saved_results.txt"))
for size, test_scores in test_scores_dict_loaded.items():
    plt.plot(x, test_scores, label="Ensemble size: {}".format(size))

plt.ylabel("Test error")
plt.xlabel("Training iterations")    
plt.legend()
plt.show()

# COMMAND ----------

def get_all_preds(iter_):
    ensemble_sizes = [1, 2, 3, 5]
    pairs = []
    
    for size in ensemble_sizes:
        for i in range(size):
            pairs.append((size, i))
    
    models = []
    
    for i,j in pairs:
        params_path = "/dbfs/VideoPose3D/saved_models/humaneva/checkpoints/supervised/{}_members_ensemble{}_iter{}.ckpt".format(i,j,iter_)
        params = torch.load(params_path, map_location=torch.device("cpu"))
        models.append(TemporalModel.from_state_dict(params, hyperparams))
        
    test_preds = get_test_predictions(models, pos2D_test, args)
    return test_preds

# COMMAND ----------

test_preds = get_all_preds(99)

# COMMAND ----------

from random import sample

def eval_last_iteration(test_preds):
    n_samples = 30
    sizes = np.arange(11)+1
    avg_test_errors = []
    test_stdvs = []
    for size in sizes:
        
        ensemble_errors = []
        
        for s in range(n_samples):
            preds_subset = sample(test_preds, k=size)
            mean_preds = torch.mean(torch.stack(preds_subset), axis=0)
            error = mpjpe(pos3D_test, mean_preds).item()
            ensemble_errors.append(error)
            
        avg_test_errors.append(sum(ensemble_errors)/len(ensemble_errors))
        test_stdvs.append(np.std(ensemble_errors))
        
    return avg_test_errors, test_stdvs
        
mean_error, stdvs  = eval_last_iteration(test_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Further analysis of test error vs. ensemble size
# MAGIC 
# MAGIC For a more torough evaluation of how testing error is affected by ensemble size, we evaluate different subsets of the 11 trained models. We evaluate the final models from the end of the trianing runs. The figure shows mean and standard deviations for test errors for ensembles of different sizes. Now we see more clearly that test error is reduced when ensemle sizes are increased. However, the standard deviation for small ensemble sizes are large. That explains why small ensembles can outperform the larger ones.

# COMMAND ----------

plt.errorbar(np.arange(11)+1, mean_error, yerr=stdvs, linestyle="none", marker="x")
plt.xlabel("Ensemble size")
plt.ylabel("Test error")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ![resutls](https://i.ibb.co/g68wZrZ/meanstdensemblesize.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualizations of predictions

# COMMAND ----------

CKPT_DIR = '/dbfs/VideoPose3D/saved_models/humaneva/checkpoints/supervised'
iter_eval = 100
ensemble_sizes = [1, 2, 3, 5]
models = []
for n_models in ensemble_sizes:
    for model_i in range(n_models):
    models.append(
        TemporalModel.from_state_dict(
                torch.load(f'{CKPT_DIR}/{n_models}_members_ensemble{model_i}_iter{iter_eval-1}.ckpt',
                       map_location=torch.device("cpu")), hyperparams))
args = Args()
pos3D_pred = get_test_predictions(models, pos2D_test, args)

# COMMAND ----------

def pflat(X):
    return X[:-1,:] / X[-1,:]

def homogenize(X):
    return np.vstack([X, np.ones((1,X.shape[1]))])

def plot_projected(pos3D_pred_list, pos3D_test, batch):
    pos3D_pred = torch.mean(torch.stack(pos3D_pred_list), axis=0)
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(131, projection='3d')
    ax2 = [fig.add_subplot(132, projection='3d'), fig.add_subplot(133, projection='3d')]
    X_test = homogenize(np.array(pos3D_test[batch, 0]).transpose())
    X_pred = homogenize(np.array(pos3D_pred[batch, 0]).transpose())
    X_pred_list = [homogenize(np.array(pose[batch, 0]).transpose()) for pose in pos3D_pred_list]
#     P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
#     x_test = P @ X_test
#     x_test = x_test[:2,:] / x_test[2,:]
#     x_pred = P @ X_pred
#     x_pred = x_pred[:2,:] / x_pred[2,:]
    ax.set_title('Ensemble')
    ax.scatter(X_test[0,:], X_test[1,:], X_test[2,:], 'k.')
    ax.scatter(X_pred[0,:], X_pred[1,:], X_pred[2,:], 'r.')
    for i in range(2):
        ax2[i].set_title(f'Model {i}')
        ax2[i].scatter(X_test[0,:], X_test[1,:], X_test[2,:], 'k.')
        ax2[i].scatter(X_pred_list[i][0,:], X_pred_list[i][1,:], X_pred_list[i][2,:], 'r.')
        
    # Plotting the skeleton
    for indices in [np.array([-1, 1, 0]),
                    np.array([1, 2, 3, 4]),
                    np.array([1, 5, 6, 7]),
                    np.array([0, 8, 9, 10]),
                    np.array([0, 11, 12, 13])]:
        ax.plot(X_pred[0,indices], X_pred[1,indices], X_pred[2,indices], 'r-')
        ax.plot(X_test[0,indices], X_test[1,indices], X_test[2,indices], 'k-')
        for i in range(2):
            ax2[i].plot(X_test[0,indices], X_test[1,indices], X_test[2,indices], 'k-')
            ax2[i].plot(X_pred_list[i][0,indices], X_pred_list[i][1,indices], X_pred_list[i][2,indices], 'r-')
    ax.view_init(elev=20., azim=-35)
    for i in range(2):
        ax2[i].view_init(elev=20., azim=-35)
    plt.show()

pos3D_pred_mean = torch.mean(torch.stack(pos3D_pred), axis=0)
error = mpjpe(pos3D_test, pos3D_pred_mean).item()
for batch_idx in [21, 5, 9]:
    plot_projected(pos3D_pred, pos3D_test, batch_idx)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional evaluation for semi-supervised models.

# COMMAND ----------

# MAGIC %md
# MAGIC ####Extracting checkpoints of semi-supervised models.

# COMMAND ----------

x = list(range(0,100,5))
x.append(99)

ensemble_sizes = [3, 5]
test_scores_dict = {}

for size in ensemble_sizes:
    test_scores_dict[size] = eval_ensemble_size_semi_supervised(size, x)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test errors during training for different ensemble sizes
# MAGIC  In our implementation, more unlabeled data is incorporated into the traninig data. It seems that using a limited number of unlabelled data helps. In our case, setting number of iteration to 40 is a proper choice. If too many unlabelled data is incorporated, the performance is degraded.

# COMMAND ----------

# save results
ensemble_sizes = [3, 5]
with open("/dbfs/VideoPose3D/semi_supervised_saved_results.txt", "w") as results_file:
    results_file.write(json.dumps(test_scores_dict))
    
for size, test_scores in test_scores_dict.items():
    plt.plot(x, test_scores, label="Ensemble size: {}".format(size))

plt.ylabel("Test error")
plt.xlabel("Training iterations")    
plt.legend()
plt.show()

 

# COMMAND ----------

x = list(range(0,100,5))
x.append(99)

test_scores_dict_loaded = json.load(open("/dbfs/VideoPose3D/semi_supervised_saved_results.txt"))
for size, test_scores in test_scores_dict_loaded.items():
    plt.plot(x, test_scores, label="Ensemble size: {}".format(size))
    plt. xlim([0, 40])

plt.ylabel("Test error")
plt.xlabel("Training iterations")    
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC  
# MAGIC ## Discussions 
# MAGIC * Simply averaging the prediciton from each ensemble is not good enough. Because the 3D pose predicitons of some models are quite unreasonable and should be ruled out.
# MAGIC * Normalizing the 2D key points. 
# MAGIC * Incoporating unlabeled data to re-train the model does not always help a lot. 