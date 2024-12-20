from src.autoencoder import AutoEncoder
from src import utils
import logging
import numpy as np
from keras.layers import Input, concatenate, Reshape
from keras.models import Model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, Callback
import os
#tf.config.set_visible_devices([],'GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0"  


class ChangeCluster(Callback):
    def __init__(self, k_dae, x_train, y_train, y_pred=None):
        super(ChangeCluster, self).__init__()
        self.k_dae = k_dae
        self.y_pred = y_pred
        self.x_train = x_train
        self.y_train = y_train

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            y_pred = self.k_dae.predict(self.x_train)
            logging.info("Result after {} epochs".format(epoch))
            utils.cluster_performance(y_pred, self.y_train)
            #print("the number of not sample change ae is {} ".format(sum(y_auto_pred == self.y_pred)))
            self.y_pred = y_pred


class KDae:
    def __init__(self, number_cluster, dataset_name='temp', k_dae_epoch=5, ae_initial_dim=(7, 5, 2,5,7),
                 initial_epoch=100, ae_dim=(7, 5, 2,5,7), epoch_ae=20, batch_size=256, save_dir='save'):
        self.number_cluster = number_cluster
        self.k_dae_epoch = k_dae_epoch
        self.ae_initial_dim = ae_initial_dim
        self.initial_epoch = initial_epoch
        self.ae_dim = ae_dim
        self.epoch_ae = epoch_ae
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.initial_ae = None
        self.initial_label = None
        self.ae_models_list = []
        self.k_dae_model = None
        self.reconstruction_errors = []

    def _initial_clustering(self, x_data):
        _, input_dim = x_data.shape
        self.initial_ae = AutoEncoder(input_dim, self.ae_initial_dim, epoch=self.initial_epoch)
        self.initial_ae.auto_encoder_model()
        embed = self.initial_ae.fit(x_data)
        k_means_initial_model = utils.k_means(embed, self.number_cluster)
        return k_means_initial_model.labels_

    def _create_combination_model(self, input_dim):
        logging.info("Create the k_dae model")
        inputs = Input(shape=(input_dim,), name='Input_layer')
        ae_list = []
        for i in range(self.number_cluster):
            ae_list.append(self.ae_models_list[i].model(inputs))
            ae_list[i] = Reshape((1, input_dim))(ae_list[i])
        out = concatenate(ae_list, axis=1, name='Output')
        model = Model(inputs=inputs, outputs=out)
        return model

    @staticmethod
    def k_dae_loss(y_true, y_pred):
        """ loss of the k_dae

        :param y_true: The output of the k_dae model np.array with shape(batch_size,self.number_cluster,input_dim)
        :param y_pred: x_data reshape to (batch_size,self.number_cluster,input_dim)
        :return:
        """
        diff = y_true - y_pred
        reconstruction_error = tf.linalg.norm(diff, axis=-1)
        min_value = tf.reduce_min(reconstruction_error, axis=-1, keepdims=True)
        return min_value
    
    def get_reconstruction_errors(self):
        return self.reconstruction_errors
    
    def fit(self, x_data, y_data=None, dataset_name='temp', save_init_label=True, is_pre_train=False):
        input_size, input_dim = x_data.shape
        if is_pre_train:
            try:
                self.initial_label = np.load(os.path.join(self.save_dir, dataset_name, 'initial_cluster.npy'))
            except FileNotFoundError as e:
                logging.warning("initial clustering didn't find, start initial AE")
                self.initial_label = self._initial_clustering(x_data)
        else:
            self.initial_label = self._initial_clustering(x_data)
        if y_data is not None:
            logging.info("########Initial Clustering Results########")
            _ = utils.cluster_performance(self.initial_label, y_data)
        if save_init_label:
            logging.info("Save the initial clustering")
            np.save(os.path.join(self.save_dir, dataset_name, 'initial_cluster'), self.initial_label) 
            _ = y_data

        for i in range(self.number_cluster):
            logging.info("model number {} create".format(i))
            self.ae_models_list.append(AutoEncoder(input_dim, self.ae_dim, epoch=self.epoch_ae, verbose=1))
            self.ae_models_list[i].auto_encoder_model()
            # train each ae with the initial clustering
            self.ae_models_list[i].fit(x_data[self.initial_label == i])
        if y_data is not None:
            y_predict = self.predict(x_data)
            logging.info("Clustering result after each ae train separately")
            _ = utils.cluster_performance(y_predict, y_data)
        check = ModelCheckpoint('k_dae_' + dataset_name, monitor='loss', save_best_only=True)
        cb = [check]
        if y_data is not None:
            cb.append(ChangeCluster(self, x_data, y_data))
        self.k_dae_model = self._create_combination_model(input_dim)
        self.k_dae_model.compile(optimizer='SGD', loss=self.k_dae_loss)
        x_repeat = np.repeat(x_data[:, np.newaxis, :], self.number_cluster, axis=1)
        #tf.config.experimental.list_physical_devices(device_type=None)
        #with tf.devices("/cpu:0"):    
        self.k_dae_model.fit(x_data, x_repeat, epochs=self.k_dae_epoch, batch_size=self.batch_size,
                             callbacks=cb)
        reconstruction_errors = self.k_dae_model.history.history['loss']
        self.reconstruction_errors.append(reconstruction_errors)

    def predict(self, x_data):
        reconstruction_norm = []
        for i in range(self.number_cluster):
            reconstruct = self.ae_models_list[i].model.predict(x_data)
            delta = x_data - reconstruct
            reconstruction_norm.append(np.linalg.norm(delta, axis=1))
        reconstruction_norm_array = np.array(reconstruction_norm)
        y_predict = np.argmin(reconstruction_norm_array, axis=0)
        return y_predict
    
    
















