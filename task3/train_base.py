import tensorflow as tf
import numpy as np
from baseT import BTConfig, BTransformer
from tqdm import tqdm

class Custom_loss(tf.losses.Loss):
    def __init__(self, reduction=tf.losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction, name)
        self.reduction = reduction
    
    def call(self, y_true, y_predict):
        outputs = tf.losses.sparse_categorical_crossentropy(y_true, y_predict)
        return outputs * self.reduction
    
    
    
class Train:
    def __init__(self, model, loss, optimizer, metric=[]):
        self.model  = model
        self._loss_fun   = loss
        self._opt    = optimizer
        self._metric = metric
        
        self.train_loss = 0
        self.val_loss   = 0
        
    def __call__(self, train_set, val_set, epochs):
        for i in tqdm(range(epochs)):
            for step, (x_batch, y_batch) in enumerate(train_set):
                self.train_block(x_batch, y_batch)
                
            for step, (x_batch, y_batch) in enumerate(val_set):
                self.val_block(x_batch, y_batch)
            
    
    @tf.function
    def train_block(self, x_batch, y_batch):
        with tf.GradientTape as tape:
            logits          = self.model(x_batch)
            loss            = self._loss_fun(y_batch, logits)
            self.train_loss = loss
            
        drevative = tape.gradient(loss, self.model.trainable_weights)
        self._opt.apply(zip(drevative, self.model.trainable_weights))
        
    @tf.function
    def val_block(self, x_batch, y_batch):
        logits        = self.model(x_batch)
        loss          = self._loss_fun(y_batch, logits)
        self.val_loss = loss
