# Databricks notebook source
# MAGIC %md
# MAGIC Distributed ensembles of deep neural networks
# MAGIC Here we provide the necessary code for creating and training a distributed ensemble of deep neural networks

# COMMAND ----------

# MAGIC %md
# MAGIC #### Assumptions
# MAGIC * The driver node fits a small subset of the data, but otherwise does not the data not fit on one node
# MAGIC * The model parameters fit in the driver node, and at least one set of model parameters fits on a worker node.
# MAGIC * Test set fits on driver node

# COMMAND ----------

# MAGIC %md
# MAGIC * How to aggregate regressed values for each keypoint:
# MAGIC   * Average results? Smoothing effect
# MAGIC   * Robust mean + discard if the predictions disagree
# MAGIC   * Change the model to predict sigma. Use sigma for the weighted mean

# COMMAND ----------

# MAGIC %md
# MAGIC #### TODO
# MAGIC * Can we do the random split of train, into labeled and unlabeled, without collecting. What if the train data is too big to collect, but yes we only collect the group names? Set a train key (labeled/unlabeled) and map by key?
# MAGIC * Do we need to collect test data? Is it possible to just collect test losses? Hecne, doing the test predictions on the workers with the trained models already there.
# MAGIC  

# COMMAND ----------

import numpy as np
import torch
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import collect_list, size, udf
from pyspark.sql.types import BooleanType

from pyspark.sql.functions import udf

from itertools import groupby
#from pyspark.ml import Pipeline
from pyspark.rdd import PipelinedRDD

from pathlib import Path
import os
import matplotlib.pyplot as plt


# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/VideoPose3D/humaneva/

# COMMAND ----------

# MAGIC %python
# MAGIC from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType
# MAGIC 
# MAGIC humaneva_train_path = "/VideoPose3D/humaneva/humaneva15_train.csv"
# MAGIC humaneva_test_path = "/VideoPose3D/humaneva/humaneva15_test.csv"
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

from pyspark.sql import functions as F

df_train = df_train.withColumn("Group", F.concat_ws(', ', "Subject", "Action", "Camera")).drop("Subject", "Action", "Camera")
df_test = df_test.withColumn("Group", F.concat_ws(', ', "Subject", "Action", "Camera")).drop("Subject", "Action", "Camera")
display(df_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Assemble feature and target columns

# COMMAND ----------

# MAGIC %py
# MAGIC from pyspark.ml.feature import VectorAssembler
# MAGIC 
# MAGIC feature_names = []
# MAGIC target_names = []
# MAGIC 
# MAGIC n_keypoints = 15
# MAGIC for i in range(n_keypoints):
# MAGIC     feature_names.append("u{}".format(i))
# MAGIC     feature_names.append("v{}".format(i))
# MAGIC     target_names.append("X{}".format(i))
# MAGIC     target_names.append("Y{}".format(i))
# MAGIC     target_names.append("Z{}".format(i))
# MAGIC     
# MAGIC feature_assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
# MAGIC target_assembler = VectorAssembler(inputCols=target_names, outputCol="targets")
# MAGIC 
# MAGIC def assemble_vectors(df):
# MAGIC     df = feature_assembler.transform(df)
# MAGIC     df = target_assembler.transform(df)
# MAGIC     df = df.drop(*feature_names).drop(*target_names)
# MAGIC     return df
# MAGIC 
# MAGIC df_train = assemble_vectors(df_train)
# MAGIC df_test = assemble_vectors(df_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Create receptive fields

# COMMAND ----------

receptive_field = 27

w = Window.orderBy("Idx").partitionBy(["Group"]).rowsBetween(Window.currentRow-receptive_field//2, Window.currentRow+receptive_field//2)

def create_receptive_fields(df):
    df = df.withColumn("feature_sequence", collect_list("features").over(w))
    df = df.withColumn("group_sequence", collect_list("Group").over(w))
    df = df.filter(size(df.group_sequence) == receptive_field)
    df = df.drop("features")
    return df

df_train_receptive = create_receptive_fields(df_train)
df_test_receptive = create_receptive_fields(df_test)

# COMMAND ----------

display(df_train_receptive)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Split training set into labeled and unlabeled based on chunks
# MAGIC Since we are exploring in semi-supervised learning, we will have both labeled and unlabeled training data. Therefore, here we randomly split the data, with respect to the group, into an unlabeled and labeled set. To have a realistic semi-supervised setting, we assume that the unlabeled training data is slighly larger than the labeled training data. The targets are droped for the unlabeled training set.

# COMMAND ----------

# MAGIC %py
# MAGIC from random import sample, seed
# MAGIC 
# MAGIC ## find right random seed to compensate for the different size of each chunk
# MAGIC seed(0) # seed 0 gives ok split
# MAGIC chunks = df_train_receptive.select("Group").distinct().collect()
# MAGIC chunks = [x["Group"] for x in chunks]
# MAGIC 
# MAGIC num_chunks = len(chunks)
# MAGIC num_unlabeled = int(num_chunks*0.6)
# MAGIC 
# MAGIC unlabeled_chunks = sample(chunks, num_unlabeled)
# MAGIC labeled_chunks = [x for x in chunks if x not in unlabeled_chunks]
# MAGIC 
# MAGIC df_train_receptive_unlabeled = df_train_receptive.filter(df_train_receptive.Group.isin(unlabeled_chunks))
# MAGIC df_train_receptive_unlabeled = df_train_receptive_unlabeled.drop("targets")
# MAGIC df_train_receptive_labeled = df_train_receptive.filter(~df_train_receptive.Group.isin(unlabeled_chunks))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Function for asserting that all elements in list are equal (not used right now but might be useful)

# COMMAND ----------

# MAGIC %py
# MAGIC 
# MAGIC def all_equal(iterable):
# MAGIC     g = groupby(iterable)
# MAGIC     return next(g, True) and not next(g, False)
# MAGIC     
# MAGIC udf_all_equal = udf(all_equal, BooleanType())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define model
# MAGIC Here we define the 3D pose estimation model with temporal convolutions and corresponding hyperparameters. Each ensemble will use this model to train on labeled data (including pseudolabels) and make predictions on unlabeled data.

# COMMAND ----------

# MAGIC %python
# MAGIC from torch import nn
# MAGIC 
# MAGIC class Args:
# MAGIC     # Data arguments
# MAGIC     num_joints = 15
# MAGIC     
# MAGIC     # Model arguments
# MAGIC     stride = 1    # chunk size to use during training
# MAGIC     epochs = 10 # 100    # number of training epochs
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
# MAGIC             num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
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

# COMMAND ----------

# MAGIC %md
# MAGIC #### Loss
# MAGIC Here we define the loss used for training and evaluation.

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
# MAGIC #### Define dataset

# COMMAND ----------

"""
class DataSet(torch.utils.data.Dataset):
    def __init__(self, pos2D, pos3D, receptive_field):
        self.pos2D = pos2D # self.pos2D: [N_1 x 15 x 2, ..., N_B x 15 x 2]
        self.pos3D = pos3D # self.pos3D: [N_1 x 15 x 3, ..., N_B x 15 x 3]
        self.receptive_field = receptive_field

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, ind):
        pos2D = self.pos2D[ind] # pos2D: N x 15 x 2
        pos3D = self.pos3D[ind] # pos3D: N x 15 x 3
        i = torch.randint(pos_3d.shape[0] - self.receptive_field + 1, [1])
        pos2D_sample = pos2D[i:i+self.receptive_field]                 # pos2D_sample: 27 x 15 x 2
        pos3D_sample = pos3D[i+(self.receptive_field - 1) // 2 ][None] # pos3D_sample:  1 x 15 x 3
        return pos2D_sample, pos3D_sample
"""

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
# MAGIC #### Train and predict methods 

# COMMAND ----------

def train(params, hyperparams, data, args):
  
    x,y = zip(*data)
    pos2D, pos3D  = torch.stack(x), torch.stack(y)

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


def predict(params, hyperparams, x):    
    model = TemporalModel.from_state_dict(params, hyperparams)
    model.eval()
    if torch.cuda.is_available():
        x = x.cuda()
        model.cuda() 
    return model(x).detach().cpu()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Train ensemble

# COMMAND ----------

"""
# Broadcast hyperparams of models
hyperparams_rdd = sc.broadcast(hyperparams)

model_params = []
for i in range(n_models):
    model = TemporalModel(*hyperparams)
    model_params.append(model.state_dict())
    
model_params_rdd = sc.parallelize(model_params)

def train_distributed(model_params, hyperparams):
    pass
    

def train_ensemble_distributed(model_params, data, hyperparams):
    pass

    for m in len(model_params.count()):
        data.mapParitions(lambda k: train_distributed(k, model_params, hyperparams))

    
"""   

# COMMAND ----------

def train_ensemble(n_models, model_params, data, hyperparams):
    
    model_data = []
    
    args = Args()
    
    assert model_params.count() == n_models
    assert len(data) == n_models, f"Lenght mismatch, lenght of data is {len(data)}, while number of models are {n_models}"
    

    
        
    models_trained = model_params.map(lambda t: train(*(t,hyperparams,data, args)))    
    
    models_params = models_trained.map(lambda t: t._1)
    train_losses = models_trained.map(lambda t: t._2)
    
    
    print(f"Training losses: {[x[1] for x in models_trained]}")
    
    return models_params, train_losses.collect()
        

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Ensemble predictions

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
# MAGIC #### Final conversion of dataframes to torch tensors

# COMMAND ----------

display(df_train_receptive_labeled)

# COMMAND ----------

### We do not have targets for unlabelled dataset
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

  

labeled_tensor = df_train_receptive_labeled.withColumn("feature_sequence", )
#unlabeled_tensor_rdd = df_train_receptive_unlabeled.rdd.map(toTensorUnlabeled)
#test_tensor_rdd = df_test_receptive.rdd.map(toTensorLabeled)


# COMMAND ----------

print(labeled_tensor_rdd.getNumPartitions())
print(unlabeled_tensor_rdd.getNumPartitions())
print(test_tensor_rdd.getNumPartitions())

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving trained models

# COMMAND ----------

# MAGIC %py
# MAGIC def save_models(models_state_dict,save_models_dir: Path ,iter: int) -> None:
# MAGIC     """
# MAGIC     Save models after training of iteration
# MAGIC     
# MAGIC     Args:
# MAGIC         models: list of state dicts of pytorch nn.Module models to be saved
# MAGIC         save_models_dir: Path to dir where models are being saved
# MAGIC         iter: iteration
# MAGIC     """
# MAGIC     
# MAGIC     # Create saving path if it does not exist
# MAGIC     save_models_dir.mkdir(parents=True, exist_ok=True)
# MAGIC     
# MAGIC     for i_model, model in enumerate(models):
# MAGIC         torch.save(model, os.path.join(save_models_dir,f"ensemble{i_model}_iter{iter}.ckpt"))
# MAGIC  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Training loop (for semi-supervised learning)
# MAGIC The hypothesis of training ensemble models in a distributed way is that we coud obtain better label estimation.Sepcifically, the prediction of the test sample is obtained by avergaing the prediciton from each memeber. Therefore, we incoporate th 

# COMMAND ----------

n_models = 2
subset_size = 1000
total_size = n_models * subset_size

def get_labeled_subset():
    # Data is loaded into driver's memory
    data = labeled_tensor_rdd.takeSample(False, total_size)
    x, y = zip(*data)
    return torch.stack(x), torch.stack(y)

def get_unlabeled_subset():
    # Data is loaded into driver's memory
    data = unlabeled_tensor_rdd.takeSample(False, total_size)
    return torch.stack(data)

def split_for_ensemble(x,y):
    '''
    Splits data so that each member acesses unique data for training
    '''
    full_size = x.shape[0]
    split_size = full_size//n_models +1
    x = torch.split(x, split_size)
    y = torch.split(y, split_size)
    
    return list(zip(x, y))

# collect test data
data_test = test_tensor_rdd.collect()
x_test, y_test = zip(*data_test)
x_test, y_test = torch.stack(x_test).detach(), torch.stack(y_test).detach()

iterations = 10 # 10

models = []
# initiate models
for i in range(n_models):
    model = TemporalModel(*hyperparams)
    models.append(model.state_dict())

# Sample a small subset of the labeled data. 
# All data is loaded into driver's memory.
x_l, y_l = get_labeled_subset()

print(f"Training distributed ensemble of {len(models)} models")

# train using only labeled data
models = train_ensemble(n_models,
                        models,
                        split_for_ensemble(x_l, y_l),
                        hyperparams)

# evaluate on test set
with torch.no_grad():
    test_preds = evaluate_avg_on_set(models, hyperparams, x_test, n_models)
    test_mpjpe = mpjpe(test_preds, y_test)
    print(f"MPJPE for test set: {test_mpjpe}")


print("Labeled training iteration finished")

test_mpjpes = []
#train_mpjpes_iteration = []
 
# train using labeled and unlabeled data
for i in range(iterations):
    
    # evaluate on test set
    test_preds = evaluate_avg_on_set(models, hyperparams, x_test, n_models)
    test_mpjpe = mpjpe(test_preds, y_test)
    test_mpjpes.append(test_mpjpe)
    
    x_ul = get_unlabeled_subset()
    
    # predict unlabeled data
    unlabeled_preds = evaluate_avg_on_set(models,
                                          hyperparams,
                                          x_ul,
                                          n_models)
    
    # Random pick a subset of trainning data
    x_l, y_l = get_labeled_subset()
    
    # concat labeled and unlabeled data
    x_cc = torch.concat([x_l, x_ul])
    y_cc = torch.concat([y_l, unlabeled_preds])
    
    # mix labeled and unlabeled data by shuffling
    idx = torch.randperm(x_cc.shape[0])
    x_cc, y_cc = x_cc[idx], y_cc[idx]
     
    print("Running semi-supervised training iteration: {}".format(i+1))
    # train using mix of labeled and pseudolabeled data
    models = train_ensemble(n_models,
                            models,
                            split_for_ensemble(x_cc, y_cc),
                            hyperparams)
    
    #train_mpjpes_iteration.append(train_mjpes)
    
    saved_models_dir = Path("saved_models/humaneva/checkpoints/semi-supervised")
    save_models(models, saved_models_dir,i)
    
    # evaluate on test set
    with torch.no_grad():
        test_preds = evaluate_avg_on_set(models, hyperparams, x_test, n_models)
        test_mpjpe = mpjpe(test_preds, y_test)
        test_mpjpes.append(test_mpjpe)
        print(f"MPJPE for test set: {test_mpjpe}")



# COMMAND ----------

# MAGIC %py
# MAGIC %matplotlib inline
# MAGIC fig = plt.figure()
# MAGIC 
# MAGIC plt.plot(test_mpjpe)
# MAGIC plt.show()

# COMMAND ----------

# MAGIC %py
# MAGIC %matplotlib inline
# MAGIC fig = plt.figure()
# MAGIC 
# MAGIC 
# MAGIC for i_model, mpjpes in enumerate(zip(*train_mpjpes_list))
# MAGIC     plt.plot(mpjpes, label=f"Ensemble {i_model}")
# MAGIC     
# MAGIC plt.legend()
# MAGIC plt.show()

# COMMAND ----------

# evaluate on test set
test_mpjpes=[]
with torch.no_grad():
    test_preds = evaluate_avg_on_set(models, hyperparams, x_test, n_models)
    test_mpjpe = mpjpe(test_preds, y_test)
    test_mpjpes.append(test_mpjpe)
    print("MPJPE for test set:")
    print(test_mpjpes)



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Training loop (for supervised baseline)
# MAGIC We first establish the baseline, where the ensemble model is trained in a distrbuted way. Specifcially, each member of the emsemble model is trained in different work node in parallel. Besides, each member is limited to access a subset of the trainining data stored in the driver node. It is a natural idea to send the same fraction of training data to the work node. However, to avoid the scenario that the work node might no have enough space to store the subset of traning data, we set the threshold for the maximum size of the data to be stored in the work node. Pratically, the size of the subset of training data is fixed to be N=1000.

# COMMAND ----------

import torch

# optional to do data partition 
def get_partitioned_rdd(input_rdd, partition_size=1000):
  
    """Partition RDD

    Args:
    input_rdd: RDD to be partitioned

    Returns:
    Partitioned RDD
    """
    return input_rdd.mapPartitions(lambda partition: partition_all(partition_size, partition))

n_models = 10
subset_size = 1000
n_iterations = 1000
total_size = n_models * subset_size

models_supervised = []
# initiate models
for i in range(n_models):
    model = TemporalModel(*hyperparams)
    models_supervised.append(model.state_dict())

test_mpjpes_supervised = []
train_mpjpes_iteration_supervised
# train using only labeled data
for iteration in range(n_iterations):
    x_l, y_l = get_labeled_subset()


    models_supervised, train_mjpes_supervised = train_ensemble(n_models, models_supervised, split_for_ensemble(x_l, y_l), hyperparams, n_epochs)
    
    train_mpjpes_iteration_supervised.append(train_mjpes_supervised)
    
    saved_models_dir = Path("saved_models/humaneva/checkpoints/supervised")
    save_models(models_supervised, saved_models_dir,epoch)

    # Ealuate on test set
    with torch.no_grad():
        test_preds_supervised = evaluate_avg_on_set(models_supervised, hyperparams, x_test, n_models)
        test_mpjpe_supervised = mpjpe(test_preds_supervised, y_test)
        test_mpjpes_supervised.append(test_mpjpe_supervised)
        print("MPJPE for test set (supervised baseline):")
        print(test_mpjpes_supervised)

# COMMAND ----------

# MAGIC %py
# MAGIC %matplotlib inline
# MAGIC fig = plt.figure()
# MAGIC 
# MAGIC plt.plot(test_mpjpe_supervised)
# MAGIC plt.show()

# COMMAND ----------

# MAGIC %py
# MAGIC %matplotlib inline
# MAGIC fig = plt.figure()
# MAGIC 
# MAGIC 
# MAGIC for i_model, mpjpes in enumerate(zip(*train_mpjpes_list_supervised))
# MAGIC     plt.plot(mpjpes, label=f"Ensemble {i_model}")
# MAGIC     
# MAGIC plt.legend()
# MAGIC plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Latest from Main

# COMMAND ----------

# MAGIC %md
# MAGIC ### To Do
# MAGIC * a figure shows that the test error is further reduced while incorporating the pseudo-labeled data.

# COMMAND ----------

# MAGIC %py
# MAGIC %matplotlib inline
# MAGIC fig = plt.figure()
# MAGIC 
# MAGIC 
# MAGIC for i_model, mpjpes in enumerate(zip(*train_mpjpes_iteration_supervised)):
# MAGIC     plt.plot(mpjpes, label=f"Ensemble {i_model}")
# MAGIC 
# MAGIC plt.title("Supervised")
# MAGIC plt.xlabel("Iteration")
# MAGIC plt.ylabel("MPJPE train loss")
# MAGIC plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# MAGIC plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Training loop for semi-supervised learning
# MAGIC The hypothesis of training ensemble models in a distributed way is that we coud obtain better target estimation for the unllablelled. Sepcifically, the prediction of the test sample is obtained by avergaing the prediciton from each memeber. Moreover, incoporating the samples with pseudo labels predicted by ensemble models into the training data is expcted to further improve the perfromace becasue more information is contained in the training data.

# COMMAND ----------

n_models = 20
subset_size = 1000
total_size = n_models * subset_size
start_unlabelled_size = 100
# collect test data
data_test = test_tensor_rdd.collect()
x_test, y_test = zip(*data_test)
x_test, y_test = torch.stack(x_test).detach(), torch.stack(y_test).detach()

iterations = 10 # 10

models = []
# initiate models
for i in range(n_models):
    model = TemporalModel(*hyperparams)
    models.append(model.state_dict())

# Sample a small subset of the labeled data. 
# All data is loaded into driver's memory.
x_l, y_l = get_labeled_subset(labeled_tensor_rdd, total_size)

print(f"Training distributed ensemble of {len(models)} models")

# train using only labeled data
#models, train_mjpes = train_ensemble(n_models,
                        #models,
                        #split_for_ensemble(x_l, y_l,n_models, total_size),
                        #hyperparams)

models, train_mjpes = train_ensemble(n_models,
                            models,
                            sample_data_for_ensemble(x_l, y_l, n_models, subset_size),
                            hyperparams)
# evaluate on test set
with torch.no_grad():
    test_preds = evaluate_avg_on_set(models, hyperparams, x_test, n_models)
    test_mpjpe = mpjpe(test_preds, y_test)
    print(f"MPJPE for test set: {test_mpjpe}")


print("Labeled training iteration finished")

test_mpjpes = []
train_mpjpes_iteration = []
 
# train using labeled and unlabeled data
for i in range(iterations):
    
    # evaluate on test set
    with torch.no_grad():
        test_preds = evaluate_avg_on_set(models, hyperparams, x_test, n_models)
        test_mpjpe = mpjpe(test_preds, y_test)
        test_mpjpes.append(test_mpjpe)
        print(f"MPJPE for test set: {test_mpjpe}")
    # use an adaptive total size for unlablled dataste
    full_size = (i+1)* start_unlabelled_size
    x_ul = get_unlabeled_subset(unlabeled_tensor_rdd, full_size)
    
    # predict unlabeled data
    unlabeled_preds = evaluate_avg_on_set(models,
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
     
    print("Running semi-supervised training iteration: {}".format(i+1))
    # train using mix of labeled and pseudolabeled data
    #models, train_mjpes = train_ensemble(n_models,
                            #models,
                            #split_for_ensemble(x_cc, y_cc, n_models, total_size),
                            #hyperparams)
                
    models, train_mjpes = train_ensemble(n_models,
                            models,
                            sample_data_for_ensemble(x_cc, y_cc, n_models, subset_size),
                            hyperparams)
    
    train_mpjpes_iteration.append(train_mjpes)
    
    saved_models_dir = Path("/dbfs/VideoPose3D/saved_models/humaneva/checkpoints/semi-supervised")
    save_models(models, saved_models_dir,i)
    
# evaluate on test set
with torch.no_grad():
    test_preds = evaluate_avg_on_set(models, hyperparams, x_test, n_models)
    test_mpjpe = mpjpe(test_preds, y_test)
    test_mpjpes.append(test_mpjpe)
    print(f"MPJPE for test set: {test_mpjpe}")



# COMMAND ----------

# MAGIC %py
# MAGIC %matplotlib inline
# MAGIC fig = plt.figure()
# MAGIC 
# MAGIC plt.plot(test_mpjpes)
# MAGIC plt.title("Semi-supervised using pseudotargets")
# MAGIC plt.xlabel("Iteration")
# MAGIC plt.ylabel("MPJPE test loss")
# MAGIC plt.show()
# MAGIC plt.close()
# MAGIC print(len(test_mpjpes))

# COMMAND ----------

# MAGIC %py
# MAGIC %matplotlib inline
# MAGIC fig = plt.figure()
# MAGIC 
# MAGIC for i_model, mpjpes in enumerate(zip(*train_mpjpes_iteration)):
# MAGIC     plt.plot(mpjpes, label=f"Ensemble {i_model}")
# MAGIC     
# MAGIC plt.title("Semi-supervised using pseudotargets")
# MAGIC plt.xlabel("Iteration")
# MAGIC plt.ylabel("MPJPE train loss")
# MAGIC plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# MAGIC plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Function for asserting that all elements in list are equal (not used right now but might be useful)

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC def all_equal(iterable):
# MAGIC     g = groupby(iterable)
# MAGIC     return next(g, True) and not next(g, False)
# MAGIC     
# MAGIC udf_all_equal = udf(all_equal, BooleanType())
# MAGIC 
# MAGIC # Broadcast hyperparams of models
# MAGIC hyperparams_rdd = sc.broadcast(hyperparams)
# MAGIC 
# MAGIC model_params = []
# MAGIC for i in range(n_models):
# MAGIC     model = TemporalModel(*hyperparams)
# MAGIC     model_params.append(model.state_dict())
# MAGIC     
# MAGIC model_params_rdd = sc.parallelize(model_params)
# MAGIC 
# MAGIC def train_distributed(model_params, hyperparams):
# MAGIC     pass
# MAGIC     
# MAGIC 
# MAGIC def train_ensemble_distributed(model_params, data, hyperparams):
# MAGIC     pass
# MAGIC 
# MAGIC     for m in len(model_params.count()):
# MAGIC         data.mapParitions(lambda k: train_distributed(k, model_params, hyperparams))
# MAGIC 
# MAGIC   
# MAGIC 
# MAGIC 
# MAGIC # optional to do data partition 
# MAGIC def get_partitioned_rdd(input_rdd, partition_size=1000):
# MAGIC   
# MAGIC     """Partition RDD
# MAGIC 
# MAGIC     Args:
# MAGIC     input_rdd: RDD to be partitioned
# MAGIC 
# MAGIC     Returns:
# MAGIC     Partitioned RDD
# MAGIC     """
# MAGIC     return input_rdd.mapPartitions(lambda partition: partition_all(partition_size, partition))

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC 
# MAGIC %python
# MAGIC def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
# MAGIC                      limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
# MAGIC     """
# MAGIC     Render an animation. The supported output modes are:
# MAGIC      -- 'interactive': display an interactive figure
# MAGIC                        (also works on notebooks if associated with %matplotlib inline)
# MAGIC      -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
# MAGIC      -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
# MAGIC      -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
# MAGIC     """
# MAGIC     plt.ioff()
# MAGIC     fig = plt.figure(figsize=(size*(1 + len(poses)), size))
# MAGIC     ax_in = fig.add_subplot(1, 1 + len(poses), 1)
# MAGIC     ax_in.get_xaxis().set_visible(False)
# MAGIC     ax_in.get_yaxis().set_visible(False)
# MAGIC     ax_in.set_axis_off()
# MAGIC     ax_in.set_title('Input')
# MAGIC 
# MAGIC     ax_3d = []
# MAGIC     lines_3d = []
# MAGIC     trajectories = []
# MAGIC     radius = 1.7
# MAGIC     for index, (title, data) in enumerate(poses.items()):
# MAGIC         ax = fig.add_subplot(1, 1 + len(poses), index+2, projection='3d')
# MAGIC         ax.view_init(elev=15., azim=azim)
# MAGIC         ax.set_xlim3d([-radius/2, radius/2])
# MAGIC         ax.set_zlim3d([0, radius])
# MAGIC         ax.set_ylim3d([-radius/2, radius/2])
# MAGIC         try:
# MAGIC             ax.set_aspect('equal')
# MAGIC         except NotImplementedError:
# MAGIC             ax.set_aspect('auto')
# MAGIC         ax.set_xticklabels([])
# MAGIC         ax.set_yticklabels([])
# MAGIC         ax.set_zticklabels([])
# MAGIC         ax.dist = 7.5
# MAGIC         ax.set_title(title) #, pad=35
# MAGIC         ax_3d.append(ax)
# MAGIC         lines_3d.append([])
# MAGIC         trajectories.append(data[:, 0, [0, 1]])
# MAGIC     poses = list(poses.values())
# MAGIC 
# MAGIC     # Decode video
# MAGIC     if input_video_path is None:
# MAGIC         # Black background
# MAGIC         all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
# MAGIC     else:
# MAGIC         # Load video using ffmpeg
# MAGIC         all_frames = []
# MAGIC         for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
# MAGIC             all_frames.append(f)
# MAGIC         effective_length = min(keypoints.shape[0], len(all_frames))
# MAGIC         all_frames = all_frames[:effective_length]
# MAGIC         
# MAGIC         keypoints = keypoints[input_video_skip:] # todo remove
# MAGIC         for idx in range(len(poses)):
# MAGIC             poses[idx] = poses[idx][input_video_skip:]
# MAGIC         
# MAGIC         if fps is None:
# MAGIC             fps = get_fps(input_video_path)
# MAGIC     
# MAGIC     if downsample > 1:
# MAGIC         keypoints = downsample_tensor(keypoints, downsample)
# MAGIC         all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
# MAGIC         for idx in range(len(poses)):
# MAGIC             poses[idx] = downsample_tensor(poses[idx], downsample)
# MAGIC             trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
# MAGIC         fps /= downsample
# MAGIC 
# MAGIC     initialized = False
# MAGIC     image = None
# MAGIC     lines = []
# MAGIC     points = None
# MAGIC     
# MAGIC     if limit < 1:
# MAGIC         limit = len(all_frames)
# MAGIC     else:
# MAGIC         limit = min(limit, len(all_frames))
# MAGIC 
# MAGIC     parents = skeleton.parents()
# MAGIC     def update_video(i):
# MAGIC         nonlocal initialized, image, lines, points
# MAGIC 
# MAGIC         for n, ax in enumerate(ax_3d):
# MAGIC             ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
# MAGIC             ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])
# MAGIC 
# MAGIC         # Update 2D poses
# MAGIC         joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
# MAGIC         colors_2d = np.full(keypoints.shape[1], 'black')
# MAGIC         colors_2d[joints_right_2d] = 'red'
# MAGIC         if not initialized:
# MAGIC             image = ax_in.imshow(all_frames[i], aspect='equal')
# MAGIC             
# MAGIC             for j, j_parent in enumerate(parents):
# MAGIC                 if j_parent == -1:
# MAGIC                     continue
# MAGIC                     
# MAGIC                 if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
# MAGIC                     # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
# MAGIC                     lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
# MAGIC                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))
# MAGIC 
# MAGIC                 col = 'red' if j in skeleton.joints_right() else 'black'
# MAGIC                 for n, ax in enumerate(ax_3d):
# MAGIC                     pos = poses[n][i]
# MAGIC                     lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
# MAGIC                                                [pos[j, 1], pos[j_parent, 1]],
# MAGIC                                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
# MAGIC 
# MAGIC             points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)
# MAGIC 
# MAGIC             initialized = True
# MAGIC         else:
# MAGIC             image.set_data(all_frames[i])
# MAGIC 
# MAGIC             for j, j_parent in enumerate(parents):
# MAGIC                 if j_parent == -1:
# MAGIC                     continue
# MAGIC                 
# MAGIC                 if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
# MAGIC                     lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
# MAGIC                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]])
# MAGIC 
# MAGIC                 for n, ax in enumerate(ax_3d):
# MAGIC                     pos = poses[n][i]
# MAGIC                     lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
# MAGIC                     lines_3d[n][j-1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
# MAGIC                     lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
# MAGIC 
# MAGIC             points.set_offsets(keypoints[i])
# MAGIC         
# MAGIC         print('{}/{}      '.format(i, limit), end='\r')
# MAGIC         
# MAGIC 
# MAGIC     fig.tight_layout()
# MAGIC     
# MAGIC     anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
# MAGIC     if output.endswith('.mp4'):
# MAGIC         Writer = writers['ffmpeg']
# MAGIC         writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
# MAGIC         anim.save(output, writer=writer)
# MAGIC     elif output.endswith('.gif'):
# MAGIC         anim.save(output, dpi=80, writer='imagemagick')
# MAGIC     else:
# MAGIC         raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
# MAGIC     plt.close()
# MAGIC     
# MAGIC print('Rendering...')    
# MAGIC input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
# MAGIC ground_truth = None
# MAGIC if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
# MAGIC     if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
# MAGIC         ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
# MAGIC if ground_truth is None:
# MAGIC     print('INFO: this action is unlabeled. Ground truth will not be rendered.')
# MAGIC 
# MAGIC gen = UnchunkedGenerator(None,
# MAGIC                          None,
# MAGIC                          [input_keypoints],
# MAGIC                          pad=pad,
# MAGIC                          causal_shift=causal_shift,
# MAGIC                          augment=args.test_time_augmentation,
# MAGIC                          kps_left=kps_left,
# MAGIC                          kps_right=kps_right,
# MAGIC                          joints_left=joints_left,
# MAGIC                          joints_right=joints_right)
# MAGIC prediction = evaluate(gen, return_predictions=True)
# MAGIC if model_traj is not None and ground_truth is None:
# MAGIC     prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
# MAGIC     prediction += prediction_traj
# MAGIC 
# MAGIC if args.viz_export is not None:
# MAGIC     print('Exporting joint positions to', args.viz_export)
# MAGIC     # Predictions are in camera space
# MAGIC     np.save(args.viz_export, prediction)
# MAGIC 
# MAGIC if args.viz_output is not None:
# MAGIC     if ground_truth is not None:
# MAGIC         # Reapply trajectory
# MAGIC         trajectory = ground_truth[:, :1]
# MAGIC         ground_truth[:, 1:] += trajectory
# MAGIC         prediction += trajectory
# MAGIC 
# MAGIC     # Invert camera transformation
# MAGIC     cam = dataset.cameras()[args.viz_subject][args.viz_camera]
# MAGIC     if ground_truth is not None:
# MAGIC         prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
# MAGIC         ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
# MAGIC     else:
# MAGIC         # If the ground truth is not available, take the camera extrinsic params from a random subject.
# MAGIC         # They are almost the same, and anyway, we only need this for visualization purposes.
# MAGIC         for subject in dataset.cameras():
# MAGIC             if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
# MAGIC                 rot = dataset.cameras()[subject][args.viz_camera]['orientation']
# MAGIC                 break
# MAGIC         prediction = camera_to_world(prediction, R=rot, t=0)
# MAGIC         # We don't have the trajectory, but at least we can rebase the height
# MAGIC         prediction[:, :, 2] -= np.min(prediction[:, :, 2])
# MAGIC 
# MAGIC     anim_output = {'Reconstruction': prediction}
# MAGIC     if ground_truth is not None and not args.viz_no_ground_truth:
# MAGIC         anim_output['Ground truth'] = ground_truth
# MAGIC 
# MAGIC     input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
# MAGIC 
# MAGIC     render_animation(input_keypoints, keypoints_metadata, anim_output,
# MAGIC                      dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
# MAGIC                      limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
# MAGIC                      input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),