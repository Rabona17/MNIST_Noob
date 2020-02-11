import tensorflow as tf
import keras
from keras import datasets
#tf.enable_eager_execution()
print(tf.__version__)

(x, y), (x_test, y_test) = datasets.mnist.load_data()

batchsz = 512
train_db = tf.data.Dataset.from_tensor_slices((x, y))

def preprocess(x, y): 
    #print(x.shape,y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x,y
train_db = train_db.shuffle(1000)
train_db = train_db.batch(batchsz)
train_db = train_db.map(preprocess)
train_db = train_db.repeat(20)


test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)

w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 512 => 256
w2, b2 = tf.Variable(tf.random.normal([256, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 256 => 256
w3, b3 = tf.Variable(tf.random.normal([256, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 256 => 256
w4, b4 = tf.Variable(tf.random.normal([256, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 256 => 256
w5, b5 = tf.Variable(tf.random.normal([256, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 256 => 10
w6, b6 = tf.Variable(tf.random.normal([256, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 256 => 256
w7, b7 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
    # 256 => 256
w8, b8 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))
step=0
for _ in range(10):
  for (x, y) in train_db:
    step+=1;
    with tf.GradientTape() as tape:
      h1 = x@w1+b1
      h1 = tf.nn.relu(h1)
      h2 = h1@w2+b2
      h2 = tf.nn.relu(h2)
      h3 = h2@w3+b3
      h3 = tf.nn.relu(h3)
      h4 = h3@w4+b4
      h4 = tf.nn.relu(h4)
      h5 = h4@w5+b5
      h5 = tf.nn.relu(h5)
      h6 = h5@w6+b6
      h6 = tf.nn.relu(h6)
      h7 = h6@w7+b7
      h7 = tf.nn.relu(h7)
      h8 = h7@w8+b8
      
      loss = tf.square(y-h8)
      loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3, w4, b4,w5,b5,w6,b6,w7,b7,w8,b8])
    for p, g in zip([w1, b1, w2, b2, w3, b3, w4, b4,w5,b5,w6,b6,w7,b7,w8,b8], grads):
      p.assign_sub(0.01 * g)
    if step %100 == 0:
              # evaluate/test
              total, total_correct = 0., 0

              for x, y in test_db:
                  # layer1.
                  h1 = x@w1+b1
                  h1 = tf.nn.relu(h1)
                  h2 = h1@w2+b2
                  h2 = tf.nn.relu(h2)
                  h3 = h2@w3+b3
                  h3 = tf.nn.relu(h3)
                  h4 = h3@w4+b4
                  h4 = tf.nn.relu(h4)
                  h5 = h4@w5+b5
                  h5 = tf.nn.relu(h5)
                  h6 = h5@w6+b6
                  h6 = tf.nn.relu(h6)
                  h7 = h6@w7+b7
                  h7 = tf.nn.relu(h7)
                  h8 = h7@w8+b8
                  # [b, 10] => [b]
                  pred = tf.argmax(h8, axis=1)
                  # convert one_hot y to number y
                  y = tf.argmax(y, axis=1)
                  # bool type
                  correct = tf.equal(pred, y)
                  # bool tensor => int tensor => numpy
                  total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                  total += x.shape[0]

              print(step, 'Acc:', total_correct/total)
