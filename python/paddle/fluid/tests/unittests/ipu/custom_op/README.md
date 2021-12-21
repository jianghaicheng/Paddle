# Add custom op for Paddle on IPU

## Add custom op in Paddle

reference

https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op_cn.html

## Write custom op for PopART

reference

https://docs.graphcore.ai/projects/popart-user-guide/en/latest/custom_ops.html

## Register custom op for Paddle on IPU

这里采用即时编译(JIT Compile) 的方法使用 custom op.

### 实现 custom op

根据上面的两个文档, 首先添加 custom op 的实现.

`leaky_relu_cpu.cc` 包含了 Paddle 中 custom op 的定义和 cpu 实现, 这里的实现是和标准的 Paddle 添加 custom op 是完全一致的. 这里的 cpu 实现不是必须的, cpu 实现可以用来检验 ipu 实现的正确性.

`leaky_relu_ipu.cc` 包含了 PopART 中 custom op 的定义和 ipu 实现, 同样的, 这里的实现和标准的 PopART 添加 custom op 是完全一致的.

### 载入 custom op

分别在 Paddle 和 PopART 中实现 custom op 的定义后, 使用 `paddle.utils.cpp_extension.load` 编译源文件并把对应的动态库加载到当前进程中.

```python

cur_dir = os.path.dirname(os.path.realpath(__file__))
custom_ops = load(
    name="custom_jit_ops",
    sources=[
        f"{cur_dir}/leaky_relu_cpu.cc",
        f"{cur_dir}/leaky_relu_ipu.cc",
    ],
    # 编译 leaky_relu_ipu.cc 时需要添加此参数
    extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'])

```

由于 Paddle 中 op 的定义和 PopART 中存在一些差异, 需要创建一个 custom op 的映射表

```python

# custom_leaky_relu is custom op type in Paddle
# LeakyRelu is custom op type in PopART
custom_ops_list = [
    compiler.IpuCustomOpIdentifier("custom_leaky_relu", "LeakyRelu"),
]

```

### 使用 custom op

```python

x = paddle.static.data(
    name=self.feed_list[0],
    shape=self.feed_shape[0],
    dtype=self.feed_dtype[0])
# custom op
out = custom_ops.custom_leaky_relu(x, **self.attrs)

```

在调用 IpuCompiler 编译计算图时, 需要将上面的 custom op 映射表作为参数传入

```python

program = compiler.IpuCompiler(
    main_prog,
    ipu_strategy=ipu_strategy,
    custom_ops=custom_ops_list).compile(feed_list, fetch_list)

```
