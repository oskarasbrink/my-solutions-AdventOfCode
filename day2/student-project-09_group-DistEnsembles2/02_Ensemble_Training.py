# Databricks notebook source
# MAGIC %md
# MAGIC # Training Distributed Ensembles for 3D Human Pose Estimation
# MAGIC In this notebook, we provide the necessary code of preparing the RDDs with pose data and creating and training a distributed ensemble of temporal CNNs for 3D pose estimation from the sequences of keypoints which will be described in more details further down.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import numpy as np
import torch

from random import sample, seed
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType, ArrayType
from pyspark.sql import Window
from pyspark.sql.functions import collect_list, size, udf
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import udf
from itertools import groupby
from pyspark.rdd import PipelinedRDD
from torch import nn

from pathlib import Path
import os
import matplotlib.pyplot as plt

ROOTDIR = 'VideoPose3D'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Some data exploration

# COMMAND ----------

# MAGIC %python
# MAGIC humaneva_train_path = f'/{ROOTDIR}/humaneva/humaneva15_train.csv'
# MAGIC humaneva_test_path = f'/{ROOTDIR}/humaneva/humaneva15_test.csv'
# MAGIC 
# MAGIC def load_data_from_csv(file_location):
# MAGIC     """Load and preprocess HumanEva data
# MAGIC     Args:
# MAGIC         file_location: file location from which to load the data
# MAGIC         
# MAGIC     Returns:
# MAGIC         df: spark DataFrame
# MAGIC     """
# MAGIC     file_type = "csv"
# MAGIC     infer_schema = "true"
# MAGIC     first_row_is_header = False
# MAGIC     delimiter = ","
# MAGIC     
# MAGIC     # Prepare a schema with column names+types
# MAGIC     schema = StructType() \
# MAGIC       .add("Idx",IntegerType(),True) \
# MAGIC       .add("Subject",StringType(),True) \
# MAGIC       .add("Action",StringType(),True) \
# MAGIC       .add("Camera",StringType(),True)
# MAGIC     for i in range(15):
# MAGIC         schema = schema.add(f"u{i}",DoubleType(),True).add(f"v{i}",DoubleType(),True)
# MAGIC     for i in range(15):
# MAGIC         schema = schema.add(f"X{i}",DoubleType(),True).add(f"Y{i}",DoubleType(),True).add(f"Z{i}",DoubleType(),True)
# MAGIC     
# MAGIC     # Load the data from file
# MAGIC     df = spark.read.csv(file_location, header=True, schema=schema, sep=',')
# MAGIC     return df
# MAGIC 
# MAGIC df_train = load_data_from_csv(humaneva_train_path)
# MAGIC df_test = load_data_from_csv(humaneva_test_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Let us take a closer look at `df_train`:

# COMMAND ----------

display(df_train.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Here,
# MAGIC * (`Xi`, `Yi`, `Zi`) are the 3D coordinates of the `i`-th keypoint
# MAGIC * (`ui`, `vi`) are the projected 2D coordinates of the corresponding keypoint
# MAGIC * (`Subject`, `Action`, `Camera`) identify the same group of frames for which we will further apply a sliding window approach
# MAGIC 
# MAGIC Let's plot the distribution of the train data and test data stratified by the action type to ensure we have enough observations in both.

# COMMAND ----------

# MAGIC %python
# MAGIC @F.udf(StringType())
# MAGIC def first_word(s, delimeter=' '):
# MAGIC     """Take the first word ouf of the sentence string."""
# MAGIC     return s.split(delimeter)[0]
# MAGIC 
# MAGIC display(df_train.withColumn("ActionType", first_word(df_train['Action'])))

# COMMAND ----------

display(df_test.withColumn("ActionType", first_word(df_test['Action'])))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Assembling features and targets
# MAGIC With VectorAssembler, we transoform the 3D keypoints (used as targets) into 45-dimensional vectors \[`X0`, `Y0`, `Z0`,...,`X14`, `Y14`, `Z14`\] and corresponding projected 2D keypoints (used as features) into 30-dimensional vectors \[`u0`, `v0`,...,`u14`, `v14`\].

# COMMAND ----------

df_train = df_train.withColumn("Group", F.concat_ws(', ', "Subject", "Action", "Camera")).drop("Subject", "Action", "Camera")
df_test = df_test.withColumn("Group", F.concat_ws(', ', "Subject", "Action", "Camera")).drop("Subject", "Action", "Camera")

feature_names = []
target_names = []
n_keypoints = 15 
for i in range(n_keypoints):
    # features correspond to 2D positions
    feature_names.append("u{}".format(i))
    feature_names.append("v{}".format(i))
    # targets correspond to 3D positions
    target_names.append("X{}".format(i))
    target_names.append("Y{}".format(i))
    target_names.append("Z{}".format(i))
    
# merge u, v into a vector column
feature_assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
# merge X, Y, Z into a vector column
target_assembler = VectorAssembler(inputCols=target_names, outputCol="targets")

def assemble_vectors(df):
    df = feature_assembler.transform(df)
    df = target_assembler.transform(df)
    df = df.drop(*feature_names).drop(*target_names)
    return df

df_train_vectors = assemble_vectors(df_train)
df_test_vectors = assemble_vectors(df_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Temporal splits

# COMMAND ----------

# MAGIC %md
# MAGIC Temporal convolutional networks use convolutional layers to slide over the time axis in the input sequences. Dilations are often employed to model long-term temporal relations. In our project, we use a a temporal CNN to predict 3D human pose from a small (27 frames) sequence of 2D keypoints, further referred to as **receptive field**. An example of such a network with receptive field **9** is shown below.

# COMMAND ----------

# MAGIC %md
# MAGIC ![temporal-cnn](https://dariopavllo.github.io/VideoPose3D/data/convolutions_anim.gif)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generating receptive fields

# COMMAND ----------

# MAGIC %md
# MAGIC So, the employed temporal CNN uses the temporal information, in particular, the 3D pose prediction of the current frame depends on several previous frames and several future frames. In case of confusion, how can we use the future frames — with the 27 receptive field we get a 0.2sec lag which is not so bad, but it's also possible to shift the convolutions so that we could use only previous frames for real-time applications. Since the data is provided per frame, to reduce the computational load of data pre-processing in the worker node, we first encapsulate any sequential 27 frames into one feature sequence. Each *feature* contains the 2D positions of the 15 joints (keypoints). Each *feature sequence* therefore consists of the 2D positions of 27 frames. The target of a sequence is the 3D pose of the middle frame. This data is used for training and evaluation instead of the individual positions.

# COMMAND ----------

receptive_field = 27

w = Window.orderBy("Idx").partitionBy(["Group"]).rowsBetween(Window.currentRow-receptive_field//2, Window.currentRow+receptive_field//2)

def create_receptive_fields(df):
    df = df.withColumn("feature_sequence", collect_list("features").over(w))
    df = df.withColumn("group_sequence", collect_list("Group").over(w))
    df = df.filter(size(df.group_sequence) == receptive_field)
    return df

df_train_receptive = create_receptive_fields(df_train_vectors).drop("features")
df_test_receptive = create_receptive_fields(df_test_vectors).drop("features")

# COMMAND ----------

# MAGIC %md
# MAGIC Visualisation of receptive field data

# COMMAND ----------

display(df_train_receptive)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <!---#### Split training set into labeled and unlabeled based on chunks
# MAGIC 
# MAGIC In the project, we are exploring semi-supervised learning with psuedolabels, which requires both labeled and unlabeled training data. However, the original HumanEva-I dataset does not provide pre-defined sets of labeled and unlabeled data. Therefore, we randomly split the data, with respect to the group, into an unlabeled and labeled set. To have a realistic semi-supervised setting, we assume that the unlabeled training data is slighly larger than the labeled training data. The targets are droped for the unlabeled training set.--->

# COMMAND ----------

# seed 0 has been chosen because it gives an ok split in terms of chunk sizes
# split dataset to labeled and unlabeled dataset, which is used in the framerwork of semi-supervised learning
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


# COMMAND ----------

# MAGIC %md
# MAGIC #### Converting dataframes to torch tensors
# MAGIC Here we create RDDs for training and test from the corresponding DataFrames to RDDs. Thereafter, we map the vectors to Tensor enable training using PyTorch.

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC #### Dataset splits
# MAGIC Here we provide functions for
# MAGIC * Train/Test split
# MAGIC * Labeled/Unlabeled split
# MAGIC * Dataset split for each member. Note that we provide two functions. The one is `split_for_ensemble`, which guarantees that each member accesses the unique data. The other is `sample_data_for_ensemble` that randomly samples the same size of training data for each memeber meaning that there might be some reused data over different members.

# COMMAND ----------

def get_labeled_subset(labeled_tensor_rdd, full_size):
    # Data is loaded into driver's memory
    data = labeled_tensor_rdd.takeSample(True, full_size)
    x, y = zip(*data)
    return torch.stack(x), torch.stack(y)

def split_for_ensemble(x,y, n_models, full_size):
    '''
    Splits data so that each member acesses unique data for training
    '''
    full_size = x.shape[0]
    split_size = full_size//n_models if full_size % n_models == 0 else full_size//n_models + 1
    x = torch.split(x, split_size)
    y = torch.split(y, split_size)
    return list(zip(x, y))

def get_unlabeled_subset(unlabeled_tensor_rdd, full_size):
    # Data is loaded into driver's memory
    data = unlabeled_tensor_rdd.takeSample(True, full_size)
    return torch.stack(data)

def sample_data_for_ensemble(x,y, n_models, subset_size):
    '''
    Randomly sample a subset of training data
    '''
    x_ = []
    y_ = []
    
    subset_size = np.amin([subset_size, x.size(0)])
    
    for i in range(n_models):
        perm = torch.randperm(x.size(0))
        idx = perm[:subset_size]
        x_.append(x[idx])
        y_.append(y[idx])
    return list(zip(x_, y_))


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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Temporal CNN Architecture

# COMMAND ----------

# MAGIC %md
# MAGIC #### Temporal CNNs for 3D pose estimation
# MAGIC Here we define the 3D pose estimation model with temporal convolutions. All members of our ensembles use the same model architecture.

# COMMAND ----------

# MAGIC %python
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
# MAGIC         print(x.shape)
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

# COMMAND ----------

# MAGIC %md
# MAGIC Below we define the hyperparameters for the architecture and training

# COMMAND ----------

class Args:
    num_joints = 15
    stride = 1    # temporal length of the prediction to use during training
    epochs = 10   # number of training epochs
    batch_size = 128     # batch size in terms of predicted frames
    dropout = 0.25    # dropout probability
    learning_rate = 0.001    # initial learning rate
    lr_decay = 0.996     # learning rate decay per epoch
    data_augmentation = True # train-time flipping
    test_time_augmentation = True # test-time flipping
    architecture = '3,3,3'    # filter widths separated by comma
    channels = 1024    # number of channels in convolution layers

args = Args()
filter_widths = [int(x) for x in args.architecture.split(',')]
receptive_field = np.prod(filter_widths) # model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
hyperparams = [args.num_joints, 2, args.num_joints, filter_widths, args.dropout, args.channels]

# COMMAND ----------

# MAGIC %md
# MAGIC #### MPJPE Loss
# MAGIC The loss used for training and evaluation is the mean per-joint postion error (MPJPE), which is the mean Euclidean distance between predicted joint postions and ground-truth joint postions
# MAGIC 
# MAGIC $$
# MAGIC \mathbf{MPJPE}(X^*, X)
# MAGIC =
# MAGIC \sum_{k=1}^K \frac{\|| X_k - X_k^*\||_2}{K}
# MAGIC $$
# MAGIC 
# MAGIC where \\(X_k \in \mathbb{R}^3\\) is the predicted 3D location of the \\(k\\)-th keypoint and \\(X_k^*\\) is its corresponding ground-truth 3D location.
# MAGIC <!-- in the camera coordinate system. -->
# MAGIC 
# MAGIC <!-- In order to evaluate \\(\mathbf{MPJPE}\\) properly, two types of projection ambiguities have to be handled.
# MAGIC 1) **absolute depth** ambiguity — normalize each pose by applying a translation that puts the skeleton root to the origin of the coordinate system.
# MAGIC 2) **depth flip** ambiguity — evaluate \\(\mathbf{MPJPE}\\) twice for the original and depth-flipped point cloud, retaining the better of the two. -->

# COMMAND ----------

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functionality for Ensemble Training

# COMMAND ----------

# MAGIC %md
# MAGIC #### Per-model training-predictions pipelines
# MAGIC The train and prediction models for each member are defined below. Note that Spark enables distribution of these functions on the work node automatically.

# COMMAND ----------

def train(params, hyperparams, pos2D, pos3D, args):
    """
    A training pipeline that every model in an ensemble performs
    Args:
        params -- model state dict (initial parameters)
        hyperparams -- hyperparameters corresponding to the model architecture
        pos2D -- inputs -- 2D receptive fields
        pos3D -- targets -- 3D poses
        args -- training parameters
    Returns:
        trained parameters in a model state dict
        training loss value
    """
    model = TemporalModel.from_state_dict(params, hyperparams)
    model.train()
    
    lr = args.learning_rate
    lr_decay = args.lr_decay
    train_data = DataSet(pos2D, pos3D)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    initial_momentum = 0.1
    final_momentum = 0.001
    
    losses_3d_train = []
    for epoch in range(args.epochs):
        epoch_loss_3d_train = 0
        N = 0
        for batch in dataloader:
            inputs_2d, inputs_3d = batch
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                model = model.cuda()
            inputs_3d[:, :, 0] = 0
            # Predict 3D poses
            predicted_3d_pos = model(inputs_2d)
            # Calcuclate MPJPE loss
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0]*inputs_3d.shape[1]
            loss_total = loss_3d_pos
            opt.zero_grad()
            loss_total.backward()
            # Make one optimization step on batch
            opt.step()
        losses_3d_train.append(epoch_loss_3d_train / N)
        print('[%d] lr %f 3d_train %f' % (
                epoch + 1,
                lr,
                losses_3d_train[-1] * 1000))
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in opt.param_groups:
            param_group['lr'] *= lr_decay
    err = mpjpe(model(pos2D.cuda()), pos3D.cuda())
    lossval = float(err.detach().cpu().numpy())
    return model.state_dict(), lossval


def predict(params, hyperparams, pos2D):
    """
    Inference pipeline that every model in an ensemble performs
    Args:
        params -- model state dict (initial parameters)
        hyperparams -- hyperparameters corresponding to the model architecture
        pos2D -- inputs -- 2D receptive fields
    """
    model = TemporalModel.from_state_dict(params, hyperparams)
    model.eval()
    if torch.cuda.is_available():
        pos2D = pos2D.cuda()
        model.cuda() 
    return model(pos2D).detach().cpu()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Parallel training
# MAGIC Below we define `train_ensemble` which enables the training of ensemble models in parallel. Note that the training is performed in work nodes.

# COMMAND ----------

def train_ensemble(n_models, model_params, data, hyperparams):
    """
    Args:
        n_models -- number of ensemble members
        model_params -- list of learnable parameters for each member
        data --  a list of training dataset for each member
        hyperparams -- hyperparameters corresponding to each member's model architecture (same for all)
    """
    model_data = []
    args = Args()
    assert len(model_params) == n_models
    assert len(data) == n_models, f"Lenght mismatch, lenght of data is {len(data)}, while number of models are {n_models}"
    
    for i, (x, y) in enumerate(data):
        # Tuples of model parameters, hyperparamers, training data, and arguments for each member.
        model_data.append((model_params[i], hyperparams, x, y, args))
    
    # create an RDD with model data
    model_data_rdd = sc.parallelize(model_data)
    # train each memeber using their own data
    models_trained = model_data_rdd.map(lambda t: train(*t))
    # send trained models state dicts and loss values to the driver node
    models_trained = models_trained.collect()
    
    # x[0] -> trained model paramteres
    # x[1] -> trainig loss value
    print(f"Training losses: {[x[1] for x in models_trained]}")
    
    return [x[0] for x in models_trained], [x[1] for x in models_trained]  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Ensemble-based predictions
# MAGIC Predictions are done on worker nodes.

# COMMAND ----------

def ensemble_predictions(models, hyperparams, test_x):
    pred_iter = _pred_models_iter(models, hyperparams, test_x)
    return pred_iter.map(lambda t: predict(*t))

def ensemble_predictions_reduced(models, hyperparams, test_x, reduce_fn):
    return ensemble_predictions(models, hyperparams, test_x).reduce(reduce_fn)

def _pred_models_iter(models, hyperparams, test_x):
    if isinstance(models, PipelinedRDD):
        return models.map(lambda model: (model, test_x))
    elif isinstance(models, list): # our case
        models_and_data = [(params, hyperparams, test_x) for params in models]
        return sc.parallelize(models_and_data)
    else:
        raise TypeError("'models' must be an RDD or a list")
        

def evaluate_avg_on_set(models, hyperparams, dataset, n_models):
    predictions_sum = ensemble_predictions_reduced(models, hyperparams, dataset, lambda x, y: x + y) # Tensor output
    predictions_avg = predictions_sum/n_models
    
    return predictions_avg  

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving the models

# COMMAND ----------

def save_models(models_state_dict, save_models_dir: Path, iter: int, n_member : int) -> None:
    """
    Save models after training of iteration
    
    Args:
        models_state_dict: list of state dicts of pytorch nn.Module models to be saved
        save_models_dir: Path to dir where models are being saved
        iter: iteration
        n_member: number of members in the current ensemble model
    """
    # Create saving path if it does not exist
    save_models_dir.mkdir(parents=True, exist_ok=True)
    
    for i_model, model_state_dict in enumerate(models_state_dict):
        torch.save(model_state_dict, os.path.join(save_models_dir, f"{n_member}_members_ensemble{i_model}_iter{iter}.ckpt"))
    print(f'Saved iter. {iter} to {save_models_dir}')
        
def save_models_with_results(models_state_dict, train_mpjpes, test_mpjpes, save_models_dir, iter, n_member):
    save_models_dir.mkdir(parents=True, exist_ok=True)
    for i_model, (model_state_dict, train_mpjpe, test_mpjpe) in \
            enumerate(zip(models_state_dict, train_mpjpes, test_mpjpes)):
        data = {
            'model_state_dict': model_state_dict,
            'train_mpjpe': train_mpjpe,
            'test_mpjpe': test_mpjpe,
        }
        torch.save(data, os.path.join(save_models_dir, f"{n_member}_members_ensemble{i_model}_iter{iter}.ckpt"))
    print(f'Saved iter. {iter} to {save_models_dir}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actual Distributed Ensemble Training

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Supervised training
# MAGIC Below we train an ensemble of models, where each model is trained in a distrbuted way. Specifically, each member is trained in a different work node in parallel. The hypothesis is that the prediction should be more accurate than the single model. Moreover, each member is limited to access a subset of the trainining data stored in the driver node. It is a natural idea to send the same fraction of training data to the work node. However, to avoid the scenario that the work node might not have enough space to store the subset of traning data, we set the threshold for the maximum size of the data to be stored in the work node. Pratically, the size of the subset of training data is fixed to be N=1000. We tried up to 11 models in an ensemble.

# COMMAND ----------

saved_models_dir = Path(f"/dbfs/{ROOTDIR}/saved_models/humaneva/checkpoints/supervised")
n_models_set = [1, 2, 3, 5, 10]
# collect test data
data_test = test_tensor_rdd.collect()
x_test, y_test = zip(*data_test)
x_test, y_test = torch.stack(x_test).detach(), torch.stack(y_test).detach()
subset_size = 1000 # subset of training data allocated to each work node.
n_iterations = 100

for n_models in n_models_set:
    test_mpjpes_supervised = []
    train_mpjpes_iteration_supervised = []
    total_size = n_models * subset_size

    for n_models in n_models_set:
        models_supervised = []
        # initiate models
        for i in range(n_models):
            model = TemporalModel(*hyperparams)
            models_supervised.append(model.state_dict())

        # train using only labeled data
        for iteration in range(n_iterations):
            x_l, y_l = get_labeled_subset(labeled_tensor_rdd, total_size)


            models_supervised, train_mjpes_supervised = train_ensemble(n_models, models_supervised, split_for_ensemble(x_l, y_l,n_models, total_size), hyperparams)

            train_mpjpes_iteration_supervised.append(train_mjpes_supervised)
            save_models(models_supervised, saved_models_dir, iteration, n_models)

            # Ealuate on test set
            with torch.no_grad():
                test_preds_supervised = evaluate_avg_on_set(models_supervised, hyperparams, x_test, n_models)
                test_mpjpe_supervised = mpjpe(test_preds_supervised, y_test)
                test_mpjpes_supervised.append(test_mpjpe_supervised)
                print("MPJPE for test set (supervised baseline):")
                print(test_mpjpes_supervised)

# evaluate on test set
test_mpjpes = []
with torch.no_grad():
    test_preds = evaluate_avg_on_set(models_supervised, hyperparams, x_test, n_models)
    test_mpjpe = mpjpe(test_preds, y_test)
    test_mpjpes.append(test_mpjpe)
    print("MPJPE for test set:")
    print(test_mpjpes)

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC #### Semi-supervised training
# MAGIC Due to the time and computational limitation, only two ensemble models including the size of memeber is 3 and 5 are conducted.

# COMMAND ----------

saved_semisupervised_models_dir = Path(f"/dbfs/{ROOTDIR}/saved_models/humaneva/checkpoints/semi_supervised")
saved_pre_trained_supervised_models_dir = Path(f"/dbfs/{ROOTDIR}/saved_models/humaneva/checkpoints/semi_supervised/pretrained")
subset_size = 500 # it seems that we have some memory issues. Decrease the subset size helps.
start_unlabelled_size = 10
iterations = 100

# collect test data
data_test = test_tensor_rdd.collect()
x_test, y_test = zip(*data_test)
x_test, y_test = torch.stack(x_test).detach(), torch.stack(y_test).detach()

# COMMAND ----------

n_models_set =[3, 5]
for n_models in n_models_set:
    total_size = n_models * subset_size
     
    ################## Initialize models ##################
    models_semisupervised = []
    for model_idx in range(n_models):
        model = TemporalModel(*hyperparams)
        models_semisupervised.append(model.state_dict())
    
     
    print(f"Supervised pre-training of {len(models_semisupervised)} models started")
    for iter in range(100):
        x_l, y_l = get_labeled_subset(labeled_tensor_rdd, total_size)
        models_semisupervised, train_mjpes = train_ensemble(n_models,
                                   models_semisupervised,
                                    sample_data_for_ensemble(x_l, y_l, n_models, subset_size),
                                    hyperparams)
        save_models(models_semisupervised,saved_pre_trained_supervised_models_dir, iter,n_models)
                                     
                                     

    ################## Semi-supervised training ##################
    print(f"Semi-supervised training of {len(models_semisupervised)} models started")
    train_mpjpes_iteration = []
    test_mpjpes_iteration = []
    # train using labeled and unlabeled data
    for iter in range(iterations):
        # use an adaptive total size for unlabelled dataste
        full_size = (iter+1) * start_unlabelled_size
        x_ul = get_unlabeled_subset(unlabeled_tensor_rdd, full_size)
        # predict unlabeled data
        unlabeled_preds = evaluate_avg_on_set(models_semisupervised,
                                              hyperparams,
                                              x_ul,
                                              n_models)

        # Random pick a subset of trainning data
        x_l, y_l = get_labeled_subset(labeled_tensor_rdd, total_size)

        # concat labeled and unlabeled data
        x_cc = torch.concat([x_l, x_ul])
        y_cc = torch.concat([y_l, unlabeled_preds])

        # mix labeled and unlabeled data by shuffling
        idx = torch.randperm(x_cc.shape[0])
        x_cc, y_cc = x_cc[idx], y_cc[idx]

        # train using mix of labeled and pseudolabeled data
        models_semisupervised, train_mjpes = train_ensemble(n_models,
                                models_semisupervised,
                                split_for_ensemble(x_cc, y_cc, n_models, total_size),
                                hyperparams)

        train_mpjpes_iteration.append(train_mjpes)  
        # evaluate on test set
        with torch.no_grad():
            test_preds = evaluate_avg_on_set(models_semisupervised, hyperparams, x_test, n_models)
            test_mpjpes = mpjpe(test_preds, y_test)
            test_mpjpes_iteration.append(test_mpjpes)
            print(f'Iteration: {iter+1}\ttrain MPJPE: {train_mjpes}\ttest MPJPE: {test_mpjpes}')
            save_models(models_semisupervised,
                                    saved_semisupervised_models_dir,
                                    iter,
                                    n_models)