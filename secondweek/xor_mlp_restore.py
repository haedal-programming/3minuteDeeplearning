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

# restore 하기 위해 saver 만들기
saver = tf.train.Saver()

# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    saver.restore(sess, 'temp/model.ckpt')

    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
    print("\n=== Test ===")
    print("X: [0, 0], Y:", sess.run(hypothesis, feed_dict={X: [[0, 0]]}))
    print("X: [0, 1], Y:", sess.run(hypothesis, feed_dict={X: [[0, 1]]}))
    print("X: [1, 0], Y:", sess.run(hypothesis, feed_dict={X: [[1, 0]]}))
    print("X: [1, 1], Y:", sess.run(hypothesis, feed_dict={X: [[1, 1]]}))