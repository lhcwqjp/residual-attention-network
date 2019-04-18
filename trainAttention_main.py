# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""

import numpy as np
import time
import os

from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
from keras.utils import np_utils
import tensorflow as tf
from tqdm import tqdm
from utils import loadData
from model.utils import EarlyStopping
from model.residual_attention_network import ResidualAttentionNetwork
from hyperparameter import HyperParams as hp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    print("start to train ResidualAttentionModel")

    X_train, Y_train, X_test, Y_test = loadData(9)

    # 将label变为向量
    ty = np_utils.to_categorical(Y_train, 2)
    ey = np_utils.to_categorical(Y_test, 2)
    print(ty.shape)
    print(ey.shape)

    print("build graph...")
    model = ResidualAttentionNetwork()

    early_stopping = EarlyStopping(limit=1)

    x = tf.placeholder(tf.float32, [None, hp.PATCH_H, hp.PATCH_W, hp.CHANNEL])
    y = tf.placeholder(tf.float32, [None, hp.NUM_CLASSES])
    is_training = tf.placeholder(tf.bool, shape=())

    output = model.attention_92(x)

    # loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,labels=y))
        tf.summary.scalar('loss', loss)
    # optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train = optimizer.minimize(loss)
    # accuracy
    with tf.name_scope("accuracy"):
        pre_score = y
        correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    # loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output+1e-7), reduction_indices=[1]))
    # train = tf.train.AdamOptimizer(1e-4).minimize(tf.reduce_mean(loss))
    with tf.name_scope("valid"):
        valid = tf.argmax(output, 1)

    print("check shape of data...")
    print("train_X: {shape}".format(shape=X_train.shape))
    print("train_y: {shape}".format(shape=Y_train.shape))

    print("start to train...")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(hp.EPOCHS):
            train_X, train_y = shuffle(X_train, Y_train, random_state=hp.RANDOM_STATE)
            # batch_train_X, batch_valid_X, batch_train_y, batch_valid_y = train_test_split(train_X, train_y, train_size=0.8, random_state=random_state)
            n_batches = train_X.shape[0] // hp.BATCH_SIZE

            # train
            train_costs = []
            train_acc = []
            for i in tqdm(range(n_batches)):
                # print(i)
                start = i * hp.BATCH_SIZE
                end = start + hp.BATCH_SIZE
                _, _acc, _loss = sess.run([train, accuracy, loss], feed_dict={x: train_X[start:end], y: ty[start:end], is_training: True})
                train_costs.append(_loss)
                train_acc.append(_acc)
            print('EPOCH: {epoch}, Training avg cost: {train_cost}, Training avg acc: {train_acc} '
                  .format(epoch=epoch, train_cost=np.mean(train_costs), train_acc=np.mean(train_acc)))

            if early_stopping.check(np.mean(train_costs)):
                print("epoch:",epoch)
                break


        # valid
        valid_costs = []
        valid_predictions = []
        valid_scores = []
        n_batches = X_test.shape[0] // hp.VALID_BATCH_SIZE
        print("test n_batch", n_batches)
        for i in range(n_batches):
            start = i * hp.BATCH_SIZE
            end = start + hp.BATCH_SIZE
            pred, valid_cost, score = sess.run([valid, loss, pre_score], feed_dict={x: X_test[start:end], y: ey[start:end], is_training: False})
            valid_predictions.extend(pred)
            valid_costs.append(valid_cost)
            valid_scores.append(pred)

        valid_predictions = np.array(valid_predictions)
        print(valid_predictions)
        y_true = np.argmax(ey, 1).astype('int32')
        print(y_true)
        print(valid_scores)
        f1_macro = f1_score(y_true, valid_predictions, average='macro')
        f1_micro = f1_score(y_true, valid_predictions, average='micro')
        accuracy = accuracy_score(y_true, valid_predictions)
        # fpr, tpr, threshold = roc_curve(y_test, y_score, drop_intermediate=False)  # 返回三个参数,不要扔掉重复tpr fpr的阈值
        # print(fpr)
        # roc_auc = auc(fpr, tpr)
        # print("roc_auc为:", roc_auc)
    #if (epoch+1) % 5 == 0:
        print('f1_score_macro cost: {f1_macro}, f1_score_micro: {f1_micro} '
              .format(f1_macro=f1_macro, f1_micro=f1_micro))
        print('Validation cost: {valid_cost}, Validation Accuracy: {accuracy} '
              .format(valid_cost=np.mean(valid_costs), accuracy=accuracy))


        print("save model...")
        saver = tf.train.Saver()
        saver.save(sess, hp.SAVED_PATH)
