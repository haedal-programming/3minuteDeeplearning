import tensorflow as tf
import pandas as pd
import numpy as np

raw_data = pd.read_csv('wine.csv')

x_data = np.asarray(raw_data.ix[:, 1:])
y_data = np.asarray(raw_data.ix[:, 0])

N = x_data.shape[0]
idx = np.arange(N)
np.random.shuffle(idx)
train_idx = idx[:int(0.9 * N)]
test_idx = idx[int(0.9 * N):]


test_x_data = x_data[test_idx]
test_y_data = y_data[test_idx]
x_data = x_data[train_idx]
y_data = y_data[train_idx]

mean = np.mean(x_data, axis=0)
std = np.std(x_data, axis=0)
x_data = (x_data - mean) / std
test_x_data = (test_x_data - mean) / std

W1 = tf.Variable(tf.random_uniform([13, 32], -1, 1))
b1 = tf.Variable(tf.random_uniform([32], -1, 1))

W2 = tf.Variable(tf.random_uniform([32, 16], -1, 1))
b2 = tf.Variable(tf.random_uniform([16], -1, 1))

W3 = tf.Variable(tf.random_uniform([16, 3], -1, 1))
b3 = tf.Variable(tf.random_uniform([3], -1, 1))

# name: 나중에 텐서보드등으로 값의 변화를 추적하거나 살펴보기 쉽게 하기 위해 이름을 붙여줍니다.
X = tf.placeholder(tf.float32, shape=[None, 13], name="X")
Y = tf.placeholder(tf.int32, shape=[None], name="Y")
Y_onehot = tf.one_hot(Y, depth=3)
print(X)
print(Y_onehot)

# X 와 Y 의 상관 관계를 분석하기 위한 가설 수식을 작성합니다.
# W 와 X 가 행렬이므로 tf.matmul을 사용했습니다.
h1 = tf.nn.relu(tf.matmul(X, W1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
h3 = tf.matmul(h2, W3) + b3
hypothesis = tf.nn.softmax(h3)

# 손실 함수를 작성합니다.
# cross entropy : 예측값과 실제값의 cross entropy를 비용(손실) 함수로 정합니다.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_onehot, logits=h3))
# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 비용을 최소화 하는 것이 최종 목표
train_op = optimizer.minimize(cost)

acc, update_op = tf.metrics.accuracy(Y, tf.argmax(hypothesis, axis=-1))

# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화를 100번 수행합니다.
    for step in range(1000):
        # sess.run 을 통해 train_op 와 cost 그래프를 계산합니다.
        # 이 때, 가설 수식에 넣어야 할 실제값을 feed_dict 을 통해 전달합니다.
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        if step % 100 == 0:
            print(step, cost_val)
    sess.run(tf.local_variables_initializer())
    a = sess.run([update_op], feed_dict={X: x_data, Y: y_data})
    print(a[0])

    sess.run(tf.local_variables_initializer())
    a = sess.run([update_op], feed_dict={X: test_x_data, Y: test_y_data})
    print(a[0])




