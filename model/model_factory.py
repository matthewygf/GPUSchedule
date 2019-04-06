"""in MegaBytes"""
import numpy as np

cnn_models = [
    'mobilenet_v1_025',
    'googlenet',
    'inception2',
    'inception3',
    'inception4',
    'alexnet',
    'vgg11',
    'vgg16',
    'vgg19',
    'resnet50',
    'resnet101',
    'resnet152'
]

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

"""
when loading model in pytorch,
model.cuda(),
cuda_init = memory_obtained_in_practice - ckpt_size,
CUDA Initialization seems to be at least using approximately ~~ 515 - 525.
For example, the below are obtained in pytorch v1.0, CUDA 10,
didn't pass any image, hence activations are excluded, no backprop, only parameter sizes:
'alexnet': 748, 'inception3': 628, 'vgg19': 1072, 'vgg16': 1048
"""
_CUDA_INIT_SIZE = 525

def estimate_gpu_utilization(size, is_cnn, gpu_mem_cap, batch_size=64, per_data_size=0.012, add_noise=True):
    """
    :param size: the model size
    :param estimate_training_util: whether to sample from gaussian distribution and add to the estimate memory size,
    as in practice, every training job can have different batch_sizes / datasets.
    :param batch_size
    :param per_data_size, default to cifar10 = 32x32x3x4 / 1,000,000
    :return: the estimate memory used in percentage, given gpu memory capacity
    # http://cs231n.github.io/convolutional-networks/#comp
    # good explanation here
    # https://datascience.stackexchange.com/questions/12649/how-to-calculate-the-mini-batch-memory-impact-when-training-deep-learning-models
    # average utilization is around 0.52
    # https://arxiv.org/abs/1901.05758
    """
    if is_cnn:
        weights_include_backpropcache = _CUDA_INIT_SIZE + (size * 3)
        batch_data = per_data_size * batch_size
        # TODO: Heuristics, alexnet activation size is used here.
        activation = 3.5
        batch_forward_mem = activation * batch_data
        batch_backward_mem = batch_forward_mem * 2
        # in practice, my 8GiB 1080 is 8118, which is 7.92 * 1024
        actual_gpu = (gpu_mem_cap-0.07)
        # assume everything is already in MGb
        at_least = (weights_include_backpropcache + batch_forward_mem + batch_backward_mem) / ( actual_gpu * 1024)
    else:
        # TODO: heuristics for language type for now.
        # too much variances :
        # 1. word embeddings
        # 2. sequence_length
        # 3. transformer
        # 4. num words
        at_least = 0.52
    if add_noise:
        mean_gauss = 0.52
        std_gauss = mean_gauss / 2.0
        return max(round(np.random.normal(mean_gauss, std_gauss),2), round(at_least+std_gauss, 2))
    else:
        return round(float(at_least),2)



