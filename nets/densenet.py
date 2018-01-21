"""Contains a variant of the densenet model definition.

Implement two kinds of networks accoring to Densely Connected Convolutional Networks.

The value of the parameter database_name of the function densenet determines which network is used.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


# add a parameter "bottleneck" in this function
def block(net, layers, growth, bottleneck, scope='block'):
    
    idx = 1
    while idx <= layers :
       
        if bottleneck :
            bnet = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1_' + str(idx))
            idx = idx + 1
        else : 
            bnet = net  
            
        tmp = bn_act_conv_drp(bnet, growth, [3, 3],
                              scope=scope + '_conv3x3_' + str(idx))
        idx = idx + 1
        
        net = tf.concat(axis=3, values=[net, tmp])
    return net



# add parameters : dataset_name, bottleneck, compression and so on
def densenet(images,
             num_classes=1001, is_training=True, 
             dataset_name = 'imagenet', layer = 32, 
             bottleneck=True,  compression=True,
	     compression_rate = 0.5,               
             growth = 24, 
             dropout_keep_prob=0.8, 
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dataset_name: set it to 'imagenet' if you want to use the network for imagenet,               otherwise set it to a name that is not 'imagenet'.
      layer: the number of layers in every dense block of the network that is not for          imagenet.
      bottleneck: specifies whether or not including bottleneck layers in our network.
      compression: specifies whether or not reducing the number of feature-maps at                  transition layer.
      compression_rate: specifies the compression factor when compression is True,                  0 < compression_rate <= 1
      growth: The number of feature maps each layer (except bottleneck layers) produces in          every dense block.
      dropout_keep_prob: the percentage of activation values that are retained.
      scope: Optional variable_scope.
      
    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    #growth = 24
    #compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
                        
            # My code goes here. And several input parameters are added.
            # There are also changes in other functions, like bn_act_conv_drp and block
            ###########This is the start line of my code###############
            
            with slim.arg_scope(densenet_arg_scope(weight_decay=0.0001)) as sc:
                
                if dataset_name == "imagenet" :
                    # Hopefully debug it later.
                    # Before Dense Block 1 : 56 x 56 x 2*growth                
                    end_point = 'input'
                    net = slim.conv2d(images, 2*growth, [7, 7], stride=2, scope=end_point+'_conv_7')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point+'_pool_2')
                    end_points[end_point] = net
   
                    # Dense Block 1 : 56 x 56 x growth              
                    end_point = 'block1'     
                    net = block(net, 12, growth, bottleneck, scope=end_point)
                    end_points[end_point] = net

                    # Convolution 1 : 56 x 56 x growth or growth*compression_rate if compression is True               
                    end_point = 'convolution1'
                
                    if compression :
                        num_outputs = reduce_dim(net)
                
                    net = bn_act_conv_drp(net, num_outputs, [1, 1], scope=end_point)
                    end_points[end_point] = net
            
                    # Average Pooling 1 : 28 x 28 x growth or growth*compression_rate if compression is True  
                    end_point = 'pool1'
                    net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
                    end_points[end_point] = net
      
                    # Dense Block 2 : 28 x 28 x growth           
                    end_point = 'block2'     
                    net = block(net, 24, growth, bottleneck, scope=end_point)
                    end_points[end_point] = net

                    # Convolution 2 : 28 x 28 x growth or growth*compression_rate if compression is True              
                    end_point = 'convolution2'
                
                    if compression :
                        num_outputs = reduce_dim(net)
                
                    net = bn_act_conv_drp(net, num_outputs, [1, 1], scope=end_point)
                    end_points[end_point] = net
            
                    # Average Pooling 2 : 14 x 14 x growth or growth*compression_rate if compression is True    
                    end_point = 'pool2'
                    net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
                    end_points[end_point] = net
  
                    # Dense Block 3 :  14 x 14 x growth             
                    end_point = 'block3'     
                    net = block(net, 48, growth, bottleneck, scope=end_point)
                    end_points[end_point] = net

                    # Convolution 3 : 14 x 14 x growth or growth*compression_rate if compression is True              
                    end_point = 'convolution3'
                
                    if compression :
                        num_outputs = reduce_dim(net)
                
                    net = bn_act_conv_drp(net, num_outputs, [1, 1], scope=end_point)
                    end_points[end_point] = net
            
                    # Average Pooling 3 : 7 x 7 x growth or growth*compression_rate if compression is True    
                    end_point = 'pool3'
                    net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
                    end_points[end_point] = net
                
                    # Dense Block 4 : 7 x 7 x growth             
                    end_point = 'block4'     
                    net = block(net, 48, growth, bottleneck, scope=end_point)
                    end_points[end_point] = net                
                
                else :
                    # Before Dense Block 1 : 32 x 32 x 16
                    end_point = 'input'
                    net = slim.conv2d(images, 16, [7, 7], stride=2, scope=end_point+'_conv_7')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point+'_pool_2')
                    net = slim.repeat(net, 12, slim.conv2d, 16, [3, 3], scope=end_point)

                    end_points[end_point] = net
   
                    # Dense Block 1 : 32 x 32 x growth             
                    end_point = 'block1'     
                    net = block(net, layer, growth, bottleneck, scope=end_point)
                    end_points[end_point] = net

                    # Convolution 1 : 32 x 32 x growth or growth*compression_rate if compression is True               
                    end_point = 'convolution1'
                
                    if compression :
                        num_outputs = reduce_dim(net)
                
                    net = bn_act_conv_drp(net, num_outputs, [1, 1], scope=end_point)
                    end_points[end_point] = net
            
                    # Average Pooling 1 : 16 x 16 x growth or growth*compression_rate if compression is True  
                    end_point = 'pool1'
                    net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
                    end_points[end_point] = net
      
                    # Dense Block 2 : 16 x 16 x growth           
                    end_point = 'block2'     
                    net = block(net, layer, growth, bottleneck, scope=end_point)
                    end_points[end_point] = net

                    # Convolution 2 : 16 x 16 x growth or growth*compression_rate if compression is True              
                    end_point = 'convolution2'
                
                    if compression :
                        num_outputs = reduce_dim(net)
                
                    net = bn_act_conv_drp(net, num_outputs, [1, 1], scope=end_point)
                    end_points[end_point] = net
            
                    # Average Pooling 2 : 8 x 8 x growth or growth*compression_rate if compression is True    
                    end_point = 'pool2'
                    net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
                    end_points[end_point] = net
  
                    # Dense Block 3 :  8 x 8 x growth             
                    end_point = 'block3'     
                    net = block(net, layer, growth, bottleneck, scope=end_point)
                    end_points[end_point] = net

                # Last Convolution   
                # imagenet :  7 x 7 x num_classes 
                # others :  8 x 8 x num_classes
                end_point = 'lastconvolution'
                net = bn_act_conv_drp(net, num_classes, [1, 1], scope=end_point)
                end_points[end_point] = net

                # Global Average Pooling £º 1 x 1 x num_classes
                end_point = 'GlobalPool'
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name=end_point)
                end_points[end_point] = net
            
                # squeeze
                end_point = 'squeeze'
                logits = tf.squeeze(net, [1, 2], name=end_point)
                end_points[end_point] = logits
            
                # classification
                end_point = "predictions"                   
                end_points[end_point] = slim.softmax(logits, scope=end_point)
                

            ###########This is the end line of my code###############

    return logits, end_points

def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc

# Add biases_initializer
def densenet_arg_scope(weight_decay=0.0001):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the densenet model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=None, biases_initializer=tf.zeros_initializer(), padding='SAME',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
