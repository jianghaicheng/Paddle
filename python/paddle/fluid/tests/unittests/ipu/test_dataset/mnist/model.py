import paddle
import paddle.static
import paddle.fluid as fluid
import paddle.fluid.layers as layers
# IPU graph compiler
import paddle.fluid.compiler as compiler
paddle.enable_static()


# static graph
class MNIST:
    def __init__(self, cfg=None, mode='train'):
        self.cfg = cfg or {}
        assert mode.lower() in ['train', 'eval', 'infer'], \
                "mode should be 'train', 'eval' or 'infer'"
        self.mode = mode.lower()
        self.optimizer = None
        self.is_loaded_weights = False
        self.is_built = False
        # TODO(yiak) : topK accuracy op will be supported soon 
        self.use_topK_Accuracy_op = False

        # build dataset and data loader

        # build network
        # self.build_model()

        # build optimizer in train mode (included in body building already)

        # initial default callbacks
        # TODO(yiakwy)

        # initial default metrics (included in body building already)
        pass

    def build_model(self):
        self._build_input()
        self._build_body()
        avg_loss = self._outputs[1]
        optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        optimizer.minimize(avg_loss)
        self.is_built = True


    # add input place holders
    def _build_input(self):
        batch_size = self.cfg.get("batch_size", None)
        if self.cfg.get("use_ipu", False):
            if batch_size is None:
                raise ValueError("Batch size must be set when running kernels on IPU devices")

        self.img = paddle.static.data(name='img', shape=[batch_size, 1, 28, 28], dtype='float32')
        if self.cfg.get("use_ipu", False):
            self.label = paddle.static.data(name='label', shape=[batch_size, 1], dtype='int32') # INT64 not supported in Poplar and Popart framework
        else:
            self.label = paddle.static.data(name='label', shape=[batch_size, 1], dtype='int64')

    def _build_body(self):
        # Note(yiakwy): flluid.nets.simple_img_conv_pool signature changed since Paddle 1.6
        # LeNet-5 arch
        conv_pool_1 = fluid.nets.simple_img_conv_pool(
                input=self.img,
                num_filters=20,
                filter_size=5,
                pool_size=2,
                pool_stride=2,
                act="relu")
        conv_pool_bn_1 = fluid.layers.batch_norm(conv_pool_1)
        conv_pool_2 = fluid.nets.simple_img_conv_pool(
            input=conv_pool_bn_1,
            num_filters=50,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act="relu")
        self.featmap = [conv_pool_2]
        self._loss()

    
    def _loss(self):
        prediction = fluid.layers.fc(input=self.featmap[0], size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=self.label)
        avg_loss = fluid.layers.mean(loss)
        if self.use_topK_Accuracy_op:
            if self.cfg.get("use_ipu", False):
                acc = fluid.layers.accuracy(input=prediction, label=self.label, soft_label=False) # Topk operator is not supported yet
            else:
                acc = fluid.layers.accuracy(input=prediction, label=self.label)
            self._outputs = [prediction, avg_loss, acc]
        else:
            # topK accuray will be calculated using numpy
            self._outputs = [prediction, avg_loss]

    @property
    def inputs(self):
        if self.mode == 'train':
            return [self.img, self.label]
        raise NotImplemented("Not implemented for {}".format(self.mode))

    @property
    def outputs(self):
        if not self.is_built:
            raise ValueError("The model is not built!")
        return self._outputs
    