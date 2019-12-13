# -*- coding: utf-8 -*-
# @Time : 19-12-06
#refer to code https://www.cnblogs.com/chenzhen0530/p/10837622.html

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import  Adam,SGD
from keras.datasets import cifar10
from keras.layers import Dense,Dropout,Flatten
from keras.models import Sequential,Model
# from keras_applications import nasnet  #keras predefined
from keras_applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint,EarlyStopping,Callback
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler,CSVLogger
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import numpy as np
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from base_model import PyramidNetBuilder
if 'tensorflow' == K.backend():
    import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config =  tf.compat.v1.ConfigProto()
config =  tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.allow_growth = True
# set_session( tf.compat.v1.Session(config=config))
set_session( tf.Session(config=config))




##用以在训练是显示学习率
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

##EvaluateBeforeTrain
class EvaluateBeforeTrain(Callback):
    def __init__(self,checkpoint):
      super(Callback,self).__init__()
      self.checkpoint=checkpoint

    def on_epoch_begin(self,epoch,logs={}):
      #evaluate validation data
      if epoch==0:
        X=self.validation_data[0]
        Y=self.validation_data[1]
        result=self.model.evaluate(X,Y)
        loss=result[0]
        acc=result[1]
        self.checkpoint.best=acc
        print('first_val_acc is:',acc)
        
def test():
  best_filepath = "./weight/keras_PyramidNet-basic-240_cifar10_best.h5"

  #construct model and load weight
  #PyramidNet
  model=PyramidNetBuilder.build(input_shape=(32,32,3), num_outputs=10, block_type='basic', alpha=240, depth=110,mode="zero-padding")
  # base_model=VGG16(include_top=False,weights=None,input_shape=(32,32,3))
  # out = base_model.layers[-1].output
  # out = Flatten()(out)
  # out  = Dense(1024, activation='relu')(out)
  # out = Dropout(0.5)(out)
  # out = Dense(512, activation='relu')(out)
  # out = Dropout(0.3)(out)
  # out = Dense(10, activation='softmax')(out)
  # model = Model(inputs=base_model.input, outputs = out)
  # model.summary()
  if os.path.exists(best_filepath):
    model.load_weights(best_filepath)
    print("have load weight")
  else:
    print(best_filepath,"is not exist")

  #load data
  ((_, _), (testX, testY)) = cifar10.load_data()
  testX = testX.astype("float") / 255.0
  #-convert the labels from integers to vectors
  lb = LabelBinarizer()
  testY = lb.fit_transform(testY)

  opt= SGD(lr=0.01,momentum=0.9)
  model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

  loss,acc= model.evaluate(testX,testY)
  print(loss,acc)




def train():

  # construct the argument parse and parse the arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-T", "--type", default="basic") 
  parser.add_argument("-A", "--alpha", default="84")
  parser.add_argument("-D", "--depth", default="110")
  
  args = parser.parse_args()

  best_filepath = "./weight/keras_PyramidNet-%s-%s-%s_cifar10_best.h5"%(args.type,args.alpha,args.depth)
  # final_save_path="./weight/keras_PyramidNet-%s-%s-%s_cifar10_final.h5"%(args.type,args.alpha,args.depth)
  csv_file="./logfile/keras_PyramidNet-%s-%s-%s_cifar10.csv"%(args.type,args.alpha,args.depth)
  plot_model_file="./plot_model/keras_PyramidNet-%s-%s-%s_cifar10.png"%(args.type,args.alpha,args.depth)
  plot_process_file="./plot/keras_PyramidNet-%s-%s-%s_cifar10_plot.png"%(args.type,args.alpha,args.depth)
  # load the training and testing data, scale it into the range [0, 1],
  # then reshape the design matrix
  print("[INFO] loading CIFAR-10 data...")
  ((trainX, trainY), (testX, testY)) = cifar10.load_data()
  trainX = trainX.astype("float") / 255.0
  testX = testX.astype("float") / 255.0
  # trainX = trainX.reshape((trainX.shape[0], 3072))
  # testX = testX.reshape((testX.shape[0], 3072))

  #data augment
  train_generator=ImageDataGenerator(horizontal_flip=True,width_shift_range=4,height_shift_range=4)


  # convert the labels from integers to vectors
  lb = LabelBinarizer()
  trainY = lb.fit_transform(trainY)
  testY = lb.fit_transform(testY)

  # initialize the label names for the CIFAR-10 dataset
  labelNames = ["airplane", "automobile", "bird", "cat", "dear", "dog", "frog", "horse", "ship", "truck"]

  # define the NASNet-A  architecture Keras
  """Table 1: CIFAR-10: 6 @ 768, 3.3M parameters"""
  # model = nasnet.NASNet(input_shape=(32,32,3),num_blocks=6,penultimate_filters=768,classes=10)
  # model = nasnet.cifar10() 

  #VGG16
  # base_model=VGG16(include_top=False,weights=None,input_shape=(32,32,3))

  # out = base_model.layers[-1].output
  # out = Flatten()(out)
  # out  = Dense(1024, activation='relu')(out)
  # out = Dropout(0.5)(out)
  # out = Dense(512, activation='relu')(out)
  # out = Dropout(0.3)(out)
  # out = Dense(10, activation='softmax')(out)
  # model = Model(inputs=base_model.input, outputs = out)
  
  #PyramidNet
  model=PyramidNetBuilder.build(input_shape=(32,32,3), num_outputs=10, block_type=args.type,alpha=int(args.alpha), depth=int(args.depth),mode="zero-padding")

  ##DNN
  # model = Sequential()
  # model.add(Dense(1024, input_shape=(3072,), activation="relu"))
  # model.add(Dense(512, activation="relu"))
  # model.add(Dense(10, activation="softmax"))
  model.summary()
  plot_model(model,plot_model_file,show_shapes=True)
  # os._exit(0)
  if os.path.exists(best_filepath):
    model.load_weights(best_filepath)
    print("have load weight")
  print(1)
 

  # train the model using optimization
  print("[INFO] training network:")
  print(best_filepath)
  # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-1)
  opt= SGD(lr=0.1,momentum=0.9)

  #set reduce lr
  # def scheduler_cosine_decay(epoch,learning_rate=0.01, decay_steps=100, alpha=0.1):
  #     if epoch==0 or epoch==1 :
  #       global global_step
  #       global new_lr
  #       global sess
  #       global F_D
  #       F_D=[]
  #       global_step = tf.Variable(tf.constant(0))  
  #       new_lr = tf.train.cosine_decay(learning_rate,global_step, decay_steps, alpha)
  #       sess = tf.Session() 
  #     lr = sess.run(new_lr,feed_dict={global_step: epoch})
  #     F_D.append(lr)
  #     return lr
  def scheduler_decay(epoch):
    lr = K.get_value(model.optimizer.lr)
    if epoch==150 or epoch==225:
      lr = lr*0.1
    return lr

  reduce_lr =LearningRateScheduler(scheduler_decay)
  # get lr as metric
  lr_metric = get_lr_metric(opt)
  #Checkpoint setting
  '''
  filename：字符串，保存模型的路径
  monitor：需要监视的值
  verbose：信息展示模式，0或1(checkpoint的保存信息，类似Epoch 00001: saving model to ...)
  save_best_only：当设置为True时，监测值有改进时才会保存当前的模型（ the latest best model according to the quantity monitored will not be overwritten）
  mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
  save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
  period：CheckPoint之间的间隔的epoch数
  '''
  checkpoint = ModelCheckpoint(filepath=best_filepath,monitor="val_acc", verbose=1,
                              save_best_only=True,mode="max",save_weights_only=True) #checkpoint的示例
  # earlystop=EarlyStopping(monitor='acc',patience=5)
  # tensorboard = TensorBoard(log_dir="log")#tensorboard的示例
  evaluateBeforeTrain=EvaluateBeforeTrain(checkpoint)
  csv_logger = CSVLogger(csv_file,append=True)
  callbacks_list = [checkpoint,reduce_lr,evaluateBeforeTrain,csv_logger]
  model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy",lr_metric])
  # H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=300, batch_size=128,callbacks=callbacks_list,verbose=1)
  H = model.fit_generator(train_generator.flow(trainX,trainY,batch_size=128), steps_per_epoch=len(trainX)//128,validation_data=(testX, testY), epochs=300,callbacks=callbacks_list,verbose=1) #不需要batch_size,由train_generator确定batch_size
  # model.save_weights(final_save_path)

  ##plot learning rate
  # plt.figure(1)  
  # plt.plot(range(100), F_D, 'r-')  
        
  # plt.show()  

  # evaluate the network
  print("[INFO] evaluating network...")
  model.load_weights(best_filepath,by_name=True)
  predictions = model.predict(testX, batch_size=32)
  print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

  # plot the training losss and accuracy
  plt.style.use("ggplot")
  plt.figure()
  l=len(H.history["loss"])
  plt.plot(np.arange(0, l), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, l), H.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, l), H.history["acc"], label="train_acc")
  plt.plot(np.arange(0, l), H.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy on CIRFAR-10")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.savefig(plot_process_file)


train()
# test()