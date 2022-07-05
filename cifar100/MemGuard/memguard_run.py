import numpy as np
np.random.seed(1000)
import imp
import keras
from keras.models import Model
import tensorflow.compat.v1 as tf
import os
import configparser
import argparse
from scipy.special import softmax
import yaml
tf.disable_eager_execution()


config_file = './../../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']


user_label_dim=100
num_classes=1



config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.3
config_gpu.gpu_options.visible_device_list = "0"

sess = tf.InteractiveSession(config=config_gpu)
sess.run(tf.global_variables_initializer())


file_path = os.path.join(root_dir, 'cifar100',  'data', 'memguard')
train_outputs1 = np.load(os.path.join(file_path, 'prediction', 'infer_train_conf_tr.npy'))
len1 = len(train_outputs1)
train_outputs2 = np.load(os.path.join(file_path, 'prediction', 'infer_train_conf_te.npy'))
len2 = len(train_outputs2)
train_outputs = np.concatenate((train_outputs1, train_outputs2))
test_outputs1 = np.load(os.path.join(file_path, 'prediction', 'infer_ref_conf.npy'))
test_outputs2 = np.load(os.path.join(file_path, 'prediction', 'infer_test_conf.npy'))
test_outputs = np.concatenate((test_outputs1, test_outputs2))
train_logits1 = np.load(os.path.join(file_path, 'prediction', 'train_logits_tr.npy'))
train_logits2 = np.load(os.path.join(file_path, 'prediction', 'train_logits_te.npy'))
train_logits = np.concatenate((train_logits1, train_logits2))
test_logits1 = np.load(os.path.join(file_path, 'prediction', 'ref_logits.npy'))
test_logits2 = np.load(os.path.join(file_path, 'prediction', 'test_logits.npy'))
test_logits = np.concatenate((test_logits1, test_logits2))
min_len = min(len(train_outputs), len(test_outputs))
print('selected number of members and non-members are: ', min(len(train_outputs), len(test_outputs)), min(len(train_logits), len(test_logits)))


f_evaluate = np.concatenate((train_outputs[:min_len], test_outputs[:min_len]))
f_evaluate_logits = np.concatenate((train_logits[:min_len], test_logits[:min_len]))
l_evaluate = np.zeros(len(f_evaluate))
l_evaluate[:min_len] = 1
print('dataset shape information: ', f_evaluate.shape, f_evaluate_logits.shape, l_evaluate.shape, min_len)

f_evaluate_origin=np.copy(f_evaluate)  #keep a copy of original one
f_evaluate_logits_origin=np.copy(f_evaluate_logits)
#############as we sort the prediction sscores, back_index is used to get back original scores#############
sort_index=np.argsort(f_evaluate,axis=1)
back_index=np.copy(sort_index)
for i in np.arange(back_index.shape[0]):
    back_index[i,sort_index[i,:]]=np.arange(back_index.shape[1])
f_evaluate=np.sort(f_evaluate,axis=1)
f_evaluate_logits=np.sort(f_evaluate_logits,axis=1)



print("f evaluate shape: {}".format(f_evaluate.shape))
print("f evaluate logits shape: {}".format(f_evaluate_logits.shape))



##########loading defense model -------------------------------------------------------------
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, concatenate
def model_defense_optimize(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Activation('softmax')(inputs_b)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model

from keras.models import load_model
defense_model = load_model(os.path.join(root_dir, 'memguard', 'purchase_MIA_model.h5'))
weights=defense_model.get_weights()
del defense_model

input_shape=f_evaluate.shape[1:]
print("Loading defense model...")
model=model_defense_optimize(input_shape=input_shape,labels_dim=num_classes)
model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])

model.set_weights(weights)
model.trainable=False



########evaluate the performance of defense's attack model on undefended data########
scores_evaluate = model.evaluate(f_evaluate_logits, l_evaluate, verbose=0)
print('evaluate loss on model:', scores_evaluate[0])
print('evaluate accuracy on model:', scores_evaluate[1])  



output=model.layers[-2].output[:,0]
c1=1.0  #used to find adversarial examples 
c2=10.0    #penalty such that the index of max score is keeped
c3=0.1
#alpha_value=0.0 

origin_value_placeholder=tf.placeholder(tf.float32,shape=(1,user_label_dim)) #placeholder with original confidence score values (not logit)
label_mask=tf.placeholder(tf.float32,shape=(1,user_label_dim))  # one-hot encode that encodes the predicted label 
c1_placeholder=tf.placeholder(tf.float32)
c2_placeholder=tf.placeholder(tf.float32)
c3_placeholder=tf.placeholder(tf.float32)

correct_label = tf.reduce_sum(label_mask * model.input, axis=1)
wrong_label = tf.reduce_max((1-label_mask) * model.input - 1e8*label_mask, axis=1)


loss1=tf.abs(output)
### output of defense classifier is the logit, when it is close to 0, the prediction by the inference is close to 0.5, i.e., random guess.
### loss1 ensures random guessing for inference classifier ###
loss2=tf.nn.relu(wrong_label-correct_label)
### loss2 ensures no changes to target classifier predictions ###
loss3=tf.reduce_sum(tf.abs(tf.nn.softmax(model.input)-origin_value_placeholder)) #L-1 norm
### loss3 ensures minimal noise addition

loss=c1_placeholder*loss1+c2_placeholder*loss2+c3_placeholder*loss3
gradient_targetlabel=K.gradients(loss,model.input)
label_mask_array=np.zeros([1,user_label_dim],dtype=np.float)
##########################################################
result_array=np.zeros(f_evaluate.shape,dtype=np.float)
result_array_logits=np.zeros(f_evaluate.shape,dtype=np.float)
success_fraction=0.0
max_iteration=300   #max iteration if can't find adversarial example that satisfies requirements
np.random.seed(1000)
for test_sample_id in np.arange(0,f_evaluate.shape[0]):
    if test_sample_id%100==0:
        print("test sample id: {}".format(test_sample_id))
    max_label=np.argmax(f_evaluate[test_sample_id,:])
    origin_value=np.copy(f_evaluate[test_sample_id,:]).reshape(1,user_label_dim)
    origin_value_logits=np.copy(f_evaluate_logits[test_sample_id,:]).reshape(1,user_label_dim)
    label_mask_array[0,:]=0.0
    label_mask_array[0,max_label]=1.0
    sample_f=np.copy(origin_value_logits)
    result_predict_scores_initial=model.predict(sample_f)
    ########## if the output score is already very close to 0.5, we can just use it for numerical reason
    if np.abs(result_predict_scores_initial-0.5)<=1e-5:
        success_fraction+=1.0
        result_array[test_sample_id,:]=origin_value[0,back_index[test_sample_id,:]]
        result_array_logits[test_sample_id,:]=origin_value_logits[0,back_index[test_sample_id,:]]
        continue
    last_iteration_result=np.copy(origin_value)[0,back_index[test_sample_id,:]]
    last_iteration_result_logits=np.copy(origin_value_logits)[0,back_index[test_sample_id,:]]
    success=True
    c3=0.1
    iterate_time=1
    while success==True: 
        sample_f=np.copy(origin_value_logits)
        j=1
        result_max_label=-1
        result_predict_scores=result_predict_scores_initial
        while j<max_iteration and (max_label!=result_max_label or (result_predict_scores-0.5)*(result_predict_scores_initial-0.5)>0):
            gradient_values=sess.run(gradient_targetlabel,feed_dict={model.input:sample_f,origin_value_placeholder:origin_value,label_mask:label_mask_array,c3_placeholder:c3,c1_placeholder:c1,c2_placeholder:c2})[0][0]
            gradient_values=gradient_values/np.linalg.norm(gradient_values)
            sample_f=sample_f-0.1*gradient_values
            result_predict_scores=model.predict(sample_f)
            result_max_label=np.argmax(sample_f)
            j+=1        
        if max_label!=result_max_label:
            if iterate_time==1:
                print("failed sample for label not same for id: {},c3:{} not add noise".format(test_sample_id,c3))
                success_fraction-=1.0
            break                
        if ((model.predict(sample_f)-0.5)*(result_predict_scores_initial-0.5))>0:
            if iterate_time==1:
                print("max iteration reached with id: {}, max score: {}, prediction_score: {}, c3: {}, not add noise".format(test_sample_id,np.amax(softmax(sample_f)),result_predict_scores,c3))
            break
        last_iteration_result[:]=softmax(sample_f)[0,back_index[test_sample_id,:]]
        last_iteration_result_logits[:]=sample_f[0,back_index[test_sample_id,:]]
        iterate_time+=1 
        c3=c3*10
        if c3>100000:
            break
    success_fraction+=1.0
    result_array[test_sample_id,:]=last_iteration_result[:]
    result_array_logits[test_sample_id,:]=last_iteration_result_logits[:]
print("Success fraction: {}".format(success_fraction/float(f_evaluate.shape[0])))


scores_evaluate = model.evaluate(result_array_logits, l_evaluate, verbose=0)
print('evaluate loss on model:', scores_evaluate[0])
print('evaluate accuracy on model:', scores_evaluate[1])

if not os.path.exists(os.path.join(file_path, 'defense_results')):
    os.makedirs(os.path.join(file_path, 'defense_results'))

np.savez(os.path.join(file_path, 'defense_results', 'purchase_shadow_defense.npz'), defense_output=result_array, defense_logits = result_array_logits, 
         tc_outputs=f_evaluate_origin)

np.save(os.path.join(file_path, 'defense_results', 'memguard_tr.npy'), result_array[:len1])
np.save(os.path.join(file_path, 'defense_results', 'memguard_te.npy'), result_array[len1:len1+len2])
np.save(os.path.join(file_path, 'defense_results', 'memguard_ref.npy'), result_array[len1+len2:len1+len2+len1])
np.save(os.path.join(file_path, 'defense_results', 'memguard_test.npy'), result_array[len1+len2+len1:])
