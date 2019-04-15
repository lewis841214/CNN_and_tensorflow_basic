# CNN and tensorflow basis
"# CNN_and_tensorflow_basic" 

這個資料夾是記錄一些在going through CNN tutor 時學到的東西
首先先說明一下之前一直沒搞懂的，tensorflow 是在創造graph，的這個graph是什麼這一回事吧!


![https://www.tensorflow.org/guide/graphs](https://www.tensorflow.org/images/tensors_flowing.gif)


首先先來說明一下這個graph是什麼意思。
就舉這個code裡面的例子為example吧


    def cnn_model_fn(inp):
        input_layer = tf.reshape(inp, [5, 28, 28 ,1])
        #convolutional
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
        #name='conv1'
        )
        #maxpooling1
        pool1=tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2,
        padding='same',
        #name='MaxPool1'
        )
        #conv2
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
        #name='conv2'
        )
        #maxpooling2
        pool2=tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2,
        padding='same',
        #name='MaxPool1'
        )
        #faltten
        #pool2_flat = tf.reshape(pool2, [batch_size, 7 * 7 * 64])
        return pool2

現在我們用這個function訂立了半個CNN的架構，那這個神經網絡的架構被架構出來了嗎?

當然是還沒，在我們執行這個function之前，這個神經網路是沒有被建構的。
那問題來了，這個神經網路要怎麼被建構呢?

這個function要帶入的值是我們的input，所以我們在tensorflow中，要使用一個place_holder去佔住input的這個空間(之後在sess.run裡面，用feed_index feed你想帶入place_holder的值)

什麼意思呢?
如果不用place_holder去佔住這個地方會發生什麼事呢?
假設我們的input為x1,x2,x3,....x1000之類的，那我們再帶入cnn_model_fn(xi)的時候，你每代一次就等於是重新呼叫了這個function，也就是說這個function裡面的架構又被你重新架構了一次。(如果你的layer有設定name的話，python會直接報錯，說你重複使用了某個name)
也就是說，我們裡面的架構，只要建立”一次"就好，那要用什麼建立呢? x1嗎?感覺也不太對，所以這邊就是用place_holder建立。
舉例來說:

這邊創造一個placeholder x並代入這個function:

    x = tf.placeholder(tf.float32, shape=(5, 784,1))
    result=cnn_model_fn( x)
    

要注意的是，這邊把從cnn_model_fn 這個function吐出來的東西叫做result
也就是說，這個graph從place_holder出發，經過cnn_model_fn裡面很多的variable之後，把得到的結果存到result裡面，那這個要怎麼跑呢?


    with tf.Session() as sess:
        sess.run(init)
        #iterator = dataset.make_one_shot_iterator()
        #next_element = iterator.get_next()
        for i in range(10):
            #temp_element=sess.run(next_element)
            #tensor flow 是在建立graph，
            temp=sess.run(result, feed_dict={x:train_data[0:5]})
            print(temp)

注意，看最下面temp=sess.run()那邊
我們要得到的東西是result，那我們要輸入x的值為x:train_data[0:5]}
那tensorflow會把這個pass進到x，然後這個流動就會從x開始，流進cnn_mode_fn裡面的很多的variable，最後到result的地方。


這就是tensorflow運作的原理(也是我之前一直沒有弄懂的地方)

除此之外，要注意個一個地方是，tf.layers.conv2d 他輸入的資料大小為，[batch_size, width, depth, channl_number] 

所以當我們拿到一組資料，長相為[number_of_data, width, depth]時，如果他是np.array,
那我們就要幫他reshape:

    train_data=train_data.reshape(55000,28,28,1)
    #最後多加一個channel_number的dimension



最後我要來記錄一下


    tf.data.Dataset.from_tensor_slices(train_data)

這個工具，這個工具真的很好用，尤其是他在處理合併 shuffle dataset等等的事情上，都有非常優異的性能。
首先，dataset1=tf.data.Dataset.from_tensor_slices(train_data)，就是根據train_data[0],[1],[2],,,把train_data 切片，再來
label=tf.data.Dataset.from_tensor_slices(train_label)
就是把label切片
接下來用zip 這個function可以把它們合併
map等等的function也很好用

    tf.data.Dataset.from_tensor_slices(train_data)
    
    
    a = { 1, 2, 3 }
    b = { 4, 5, 6 }
    c = { (7, 8), (9, 10), (11, 12) }
    d = { 13, 14 }
    
    # The nested structure of the `datasets` argument determines the
    # structure of elements in the resulting dataset.
    Dataset.zip((a, b)) == { (1, 4), (2, 5), (3, 6) }
    Dataset.zip((b, a)) == { (4, 1), (5, 2), (6, 3) }

那要怎麼呼叫dataset裡面的東西?

    dataset=tf.data.Dataset.from_tensor_slices(train_data)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    
    with tf.Session() as sess:
        temp=sess.run(next_element)
        
        

用sess run把next_element的東西取出來，得到一個np.array
然後再用feed_dict餵進placeholder就可以啦~

