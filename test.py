import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
tf.enable_eager_execution()
from losses import grasp_loss_bt
test_func = grasp_loss_bt(2)
# 2, 2, 6
x = [   #I1
        [   [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0]  ],
        #I2
        [   [1, 2, 3, 4, 0, 0], [1, 2, 3, 0, 0, 0]  ]
    ] 
y = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]  
F = test_func(K.variable(x), K.variable(y))
p = K.eval(F)
print(len(p))
print(p)
tf.print(p)