"""in MegaBytes"""
model_sizes = {
    #https://s3.amazonaws.com/opennmt-models/onmt_esfritptro-4-1000-600_epoch13_3.12_release_v2.t7
    '4_layers_brnn': 1300,
    #https://s3.amazonaws.com/opennmt-models/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt
    'transformer': 1100,
    #https://s3.amazonaws.com/opennmt-models/ada6_bridge_oldcopy_tagged_larger_acc_54.84_ppl_10.58_e17.pt
    '1_layer_bilstm_opennmt': 900,
    #https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    'BERT_Chinese': 350,
    #https://s3.amazonaws.com/opennmt-models/gigaword_copy_acc_51.78_ppl_11.71_e20.pt
    '2_layers_lstm_gigaword': 330,
    #http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz
    'mobilenet_v1_025': 15,
    #http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
    'googlenet': 26, 
    #http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz
    'inception2': 43,
    #http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    'inception3': 104,
    #http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
    'inception4': 176,
    #http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
    'alexnet': 233,
    #https://download.pytorch.org/models/vgg11-bbd30ac9.pth
    'vgg11': 519,
    #http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
    'vgg19': 549,
    #http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
    'vgg16': 528,
    #http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
    'resnet50': 97,
    #http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz
    'resnet101': 555,
    #http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
    'resnet152': 737
    #TODO: MORE !
}

#TODO: MODELMEMORY_SIZE!!!!!!

