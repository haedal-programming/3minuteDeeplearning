import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

W1 = tf.Variable(tf.random_uniform([2, 4], -1, 1, seed = 0))
b1 = tf.Variable(tf.random_uniform([4], -1, 1, seed = 0))

W2 = tf.Variable(tf.random_uniform([4, 1], -1, 1, seed = 0))
b2 = tf.Variable(tf.random_uniform([1], -1, 1, seed = 0))

# name: 나중에 텐서보드등으로 값의 변화를 추적하거나 살펴보기 쉽게 하기 위해 이름을 붙여줍니다.
X = tf.placeholder(tf.float32, shape=[None, 2], name="X")
Y = tf.placeholder(tf.float32, shape=[None, 1], name="Y")
print(X)
print(Y)

# X 와 Y 의 상관 관계를 분석하기 위한 가설 수식을 작성합니다.
# W 와 X 가 행렬이므로 tf.matmul을 사용했습니다.
h1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
h2 = tf.matmul(h1, W2) + b2
hypothesis = tf.nn.sigmoid(h2)

# 손실 함수를 작성합니다.
# cross entropy : 예측값과 실제값의 cross entropy를 비용(손실) 함수로 정합니다.
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h2))
# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 비용을 최소화 하는 것이 최종 목표
train_op = optimizer.minimize(cost)

# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화를 20000번 수행합니다.
    for step in range(20000):
        # sess.run 을 통해 train_op 와 cost 그래프를 계산합니다.
        # 이 때, 가설 수식에 넣어야 할 실제값을 feed_dict 을 통해 전달합니다.
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        if step % 100 == 0:
            print(step, cost_val)

    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
    print("\n=== Test ===")
    print("X: [0, 0], Y:", sess.run(hypothesis, feed_dict={X: [[0, 0]]}))
    print("X: [0, 1], Y:", sess.run(hypothesis, feed_dict={X: [[0, 1]]}))
    print("X: [1, 0], Y:", sess.run(hypothesis, feed_dict={X: [[1, 0]]}))
    print("X: [1, 1], Y:", sess.run(hypothesis, feed_dict={X: [[1, 1]]}))