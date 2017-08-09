# MXNET_notes

[tutorial-1](http://mxnet.io/tutorials/index.html)

[tutorial-2](http://mxnet-bing.readthedocs.io/en/latest/tutorials/index.html)



* 准备数据
    ```python
    traindata = ...
    testdata = ...
    ```

* 构建网络

    ```python
    net = mx.sym.Variable(...)
    net = mx.sym.FullyConnected(...)
    net = mx.sym.Connvolution(...)
    net = mx.sym.Activation(...)
    net = mx.sym.LRN(...)
    net = mx.sym.Pooling(...)
    net = mx.sym.Dropout(...)
    net = mx.sym.BatchNorm(...)
    ...
    ```


* 构建模型
    ```python
    model = mx.mod.Module(symbol=net, context=mx.gpu())
    ```

* 准备train_iter 和 eval_iter
    * train_iter
        ```python
        train_iter = mx.io.NDArrayIter(data=mnist['train_data'],
                                       label=mnist['train_label'], batch_size=batch_size, 
                                       shuffle=True)
        ```
    * eval_iter
        ```test_iter
        test_iter = mx.io.NDArrayIter(data=mnist['test_data'],
                                      label=mnist['test_label'],
                                      batch_size=batch_size, 
                                      shuffle=True)
        ```

* train
    ```python
    modle.fit(train_data=train_iter,
              eval_data=eval_iter,
              eval_metric='...',
              optimizer='...'
              optimizer_params={'learning_rate':0.1},
              num_epoch=...
              )
    ```

* test
    * 准备测试数据迭代器
        ```python
        test_iter = mx.io.NDArrayIter(data=mnist['test_data'], 
                                  label=mnist['test_label'], 
                                  batch_size=batch_size, 
                                  shuffle=False)
        ```

    * 测试
        ```python
        # 回归的情况
        mse = mx.metric.MSE()

        # 分类的情况
        acc = mx.metric.Accuracy()
        lenet_model.score(test_iter, eval_metric=acc)
        print(acc)
        ```
* checkpoint
    * save checkpoint

        ```python
        model_prefix = 'lenet_checkpoint/mx_lenet' # path and name
        checkpoint = mx.callback.do_checkpoint(model_prefix)


        # 每个epoch都保存一次checkpoint
        lenet_model.fit(...,
                        epoch_end_callback=checkpoint)
        ```
    * load checkpoint
        ```python
        sym, arg_params, aux_params = mx.model.load_checkpoint('lenet_checkpoint/mx_lenet', 20)
        ```

* fine tune
    * 直接使用原来的模型进行训练
        ```python
        # load checkpoint
        sym, arg_params, aux_params = mx.model.load_checkpoint('lenet_checkpoint/mx_lenet', 20)

        lenet_model.fit(train_data=train_iter,arg_params=arg_params, aux_params=aux_params,
                eval_data=eval_iter, 
                optimizer='sgd', 
                num_epoch=10 ,
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback=mx.callback.Speedometer(batch_size, 128),
                )
        ```
    * fine tune 最后一层
    ```python
    # load checkpoint
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', epoch=0)

    # 找出倒数第二个输出层的name
    all_layers = sym.get_internals()
    all_layers.list_outputs()[-10:]

    # 接入一个新的全连接层，并且输出
    net = all_layers['flatten0_output']
    net = mx.sym.FullyConnected(data=net, num_hidden=10, name='fc')
    net = mx.sym.SoftmaxOutput(data=net, name='softmax')

    # module
    fine_tune_model = mx.mod.Module(symbol=net, context=mx.gpu())

    # train
    fine_tune_model.fit(train_data=train_iter, 
                    eval_data=eval_iter, 
                    optimizer='sgd', 
                    num_epoch=20, 
                    batch_end_callback=mx.callback.Speedometer(batch_size, 128))
    ```