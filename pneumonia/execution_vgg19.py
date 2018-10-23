import luigi

from pneumonia.model.vgg19 import VGG19


class RunMultipleModels(luigi.WrapperTask):
    def requires(self):
        
        ########################## HYPER PARAMETRO DEFINIDO: frozen_layers=21    ########################################
        # Variando estratégia de sampling
        #yield VGG19(frozen_layers=21, sampling_strategy='undersample', batch_size=64, val_batch_size=128)
        
        
        # Variando DROPOUT
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', dropout=0.1, batch_size=64, val_batch_size=128)        
        #  yield VGG19(frozen_layers=21, sampling_strategy='oversample', dropout=0.3, batch_size=64, val_batch_size=128)
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', dropout=0.5, batch_size=64, val_batch_size=128)
        
        
        # Variando kernel_initializer        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_normal', dropout=0.1)
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_normal', dropout=0.2)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_uniform', dropout=0.1)
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_uniform', dropout=0.2)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_uniform', dropout=0.1)
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_uniform', dropout=0.2)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_normal', dropout=0.1)
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_normal', dropout=0.2)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1)
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_normal', dropout=0.1)
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_normal', dropout=0.2)
        
        #Variando camada densa
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 2, all_dropout = True, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 2, all_dropout = False, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 3, all_dropout = True, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 3, all_dropout = False, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_uniform', dropout=0.1, total_dense_layers = 2, all_dropout = True, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_uniform', dropout=0.1, total_dense_layers = 2, all_dropout = False, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = True, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = False, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_normal', dropout=0.2, total_dense_layers = 2, all_dropout = True, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_normal', dropout=0.2, total_dense_layers = 2, all_dropout = False, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_normal', dropout=0.2, total_dense_layers = 3, all_dropout = True, dense_neurons = 512)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_normal', dropout=0.2, total_dense_layers = 3, all_dropout = False, dense_neurons = 512)
        
        # Variando interpolação
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_normal', dropout=0.2, total_dense_layers = 3, all_dropout = False, dense_neurons = 512, interpolation="lanczos")
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_normal', dropout=0.2, total_dense_layers = 3, all_dropout = False, dense_neurons = 512, interpolation="box")
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 1, all_dropout = False, dense_neurons = 1024, interpolation="bicubic")
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 1, all_dropout = False, dense_neurons = 1024, interpolation="lanczos")
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 1, all_dropout = False, dense_neurons = 1024, interpolation="box")
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_uniform', dropout=0.1, total_dense_layers = 1, all_dropout = False, dense_neurons = 1024, interpolation="bicubic")
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_uniform', dropout=0.1, total_dense_layers = 1, all_dropout = False, dense_neurons = 1024, interpolation="lanczos")
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='he_uniform', dropout=0.1, total_dense_layers = 1, all_dropout = False, dense_neurons = 1024, interpolation="box")
        
        # Params chico
        #yield VGG19(frozen_layers=21,input_shape=(224, 224), sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='glorot_uniform', total_dense_layers = 2, all_dropout = False, dense_neurons = 4096,dropout=None, interpolation="lanczos", learning_rate=5e-05, batch_normalization_after_vgg=False,batch_normalization_between_dense=False)
        
        #yield VGG19(frozen_layers=21,input_shape=(224, 224), sampling_strategy='oversample', batch_size=80, val_batch_size=100, kernel_initializer='glorot_uniform', total_dense_layers = 2, all_dropout = False, dense_neurons = 4096,dropout=None, interpolation="lanczos", learning_rate=5e-05, batch_normalization_after_vgg=False,batch_normalization_between_dense=False)
        
        
        # Variandooooo
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 512, interpolation="lanczos", learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = False, dense_neurons = 512, interpolation="lanczos", learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 3, all_dropout = False, dense_neurons = 512, interpolation="lanczos", learning_rate=5e-05)
        
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos", learning_rate=5e-05)
        
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 2048, interpolation="lanczos",  learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = False, dense_neurons = 2048, interpolation="lanczos", learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 3, all_dropout = False, dense_neurons = 2048, interpolation="lanczos",  learning_rate=5e-05)
        
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = True, dense_neurons = 512, interpolation="lanczos",  learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = True, dense_neurons = 512, interpolation="lanczos",  learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 3, all_dropout = True, dense_neurons = 512, interpolation="lanczos",  learning_rate=5e-05)
        
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = True, dense_neurons = 1024, interpolation="lanczos", learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = True, dense_neurons = 1024, interpolation="lanczos",learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 3, all_dropout = True, dense_neurons = 1024, interpolation="lanczos", learning_rate=5e-05)
        
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = True, dense_neurons = 2048, interpolation="lanczos",  learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = True, dense_neurons = 2048, interpolation="lanczos",  learning_rate=5e-05)
                
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.2, total_dense_layers = 3, all_dropout = True, dense_neurons = 2048, interpolation="lanczos",  learning_rate=5e-05)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = True, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 4, all_dropout = True, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=56, val_batch_size=112, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 2, all_dropout = True, dense_neurons = 2048, interpolation="lanczos",  learning_rate=5e-05)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=56, val_batch_size=112, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 2, all_dropout = False, dense_neurons = 2048, interpolation="lanczos",  learning_rate=5e-05)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=1e-05)        
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=1e-05)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05)
        
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, batch_normalization_between_dense=True)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=1e-05, batch_normalization_between_dense=True)        
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=1e-05, batch_normalization_between_dense=True)
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, batch_normalization_between_dense=True)
        
        
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, input_shape=(224, 224))
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=1e-05, input_shape=(224, 224))      
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=1e-05, input_shape=(224, 224))
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, input_shape=(224, 224))
        
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, batch_normalization_between_dense=True, input_shape=(224, 224))
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=1e-05, batch_normalization_between_dense=True, input_shape=(224, 224))      
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=1e-05, batch_normalization_between_dense=True, input_shape=(224, 224))
        
        #yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 4, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, batch_normalization_between_dense=True, input_shape=(224, 224))
        
        
        yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, batch_normalization_between_dense=True, input_shape=(256, 256), retrain_optimizer_extra_params=dict(decay=1e-6, momentum=0.9, nesterov=True),data_augmentation_params=dict(rotation_range=15, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2), data_augmentation_width_resize_range=0.1)
        
        yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, batch_normalization_between_dense=True, input_shape=(256, 256), retrain_optimizer_extra_params=dict(decay=1e-6, momentum=0.9, nesterov=True),data_augmentation_params=dict(rotation_range=15, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2), data_augmentation_width_resize_range=0.1)
        
        
        
        yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, batch_normalization_between_dense=True, input_shape=(256, 256), retrain_optimizer_extra_params=dict(decay=1e-6, momentum=0.9, nesterov=True),data_augmentation_params=dict(rotation_range=15, width_shift_range=0.25, height_shift_range=0.25, zoom_range=0.25), data_augmentation_width_resize_range=0.1)
        
        yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, batch_normalization_between_dense=True, input_shape=(256, 256), retrain_optimizer_extra_params=dict(decay=1e-6, momentum=0.9, nesterov=True),data_augmentation_params=dict(rotation_range=15, width_shift_range=0.25, height_shift_range=0.25, zoom_range=0.25), data_augmentation_width_resize_range=0.1)
        
        
        
        yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=None, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, batch_normalization_between_dense=True, input_shape=(256, 256), retrain_optimizer_extra_params=dict(decay=1e-6, momentum=0.9, nesterov=True),data_augmentation_params=dict(rotation_range=25, width_shift_range=0.25, height_shift_range=0.2, zoom_range=0.2), data_augmentation_width_resize_range=0.1)
        
        yield VGG19(frozen_layers=21, sampling_strategy='oversample', batch_size=64, val_batch_size=128, kernel_initializer='lecun_uniform', dropout=0.1, total_dense_layers = 3, all_dropout = False, dense_neurons = 1024, interpolation="lanczos",  learning_rate=5e-05, batch_normalization_between_dense=True, input_shape=(256, 256), retrain_optimizer_extra_params=dict(decay=1e-6, momentum=0.9, nesterov=True),data_augmentation_params=dict(rotation_range=25, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2), data_augmentation_width_resize_range=0.1)
        
        
        
        
        
        
        
        
        
        
