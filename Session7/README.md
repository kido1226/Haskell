# build a RNN model
## definition of model
```
data Model = Model
  { emb :: Embedding,
    rnn :: RnnParams,
    initialStates :: InitialStatesParams,
    mlp :: MLP
  }
  deriving (Generic, Parameterized)
```
## training cord
```
  (trainedModel, losses) <- foldLoop (initModel, []) numIters $ \(state, losses) i -> do
    -- forward
    let Model {..} = state
    let embedded = embedding' (toDependent $ wordEmbedding emb) inputTensor
    let h0 = toDependent (h0s initialStates)
    let (rnnOut, _) = RNN.rnnLayers rnn Tanh Nothing h0 embedded
    print rnnOut
    let meanOut = meanDim (Dim 1) KeepDim Float rnnOut
    print $ shape meanOut
    let pred = mlpForward mlp meanOut

    -- loss
    print pred
    print $ shape pred
    let loss = mseLoss ratings' pred
    let loss' = asValue loss :: Float
    when (i `mod` 10 == 0) $ do
      putStrLn $ "Iteration " ++ show i ++ " | Loss: " ++ show loss'
    -- パラメータ更新
    (newState, _) <- runStep state optimizer loss rate
    return (newState, loss' : losses)
```
I implemented as above, but the following runtime error occurred and I could not train.
```
Cannot show stacktrace
Exception: Tensors must have same number of dimensions: got 2 and 1
Exception raised from check_cat_shape_except_dim at ../aten/src/ATen/native/TensorShape.h:12 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x6b (0x2aaaab50c7ab in /lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0xce (0x2aaaab50815e in /lib/libc10.so)
frame #2: at::native::_cat_out_cpu(c10::ArrayRef<at::Tensor>, long, at::Tensor&) + 0xae6 (0x2aaaacb0f596 in /lib/libtorch_cpu.so)
frame #3: at::native::_cat_cpu(c10::ArrayRef<at::Tensor>, long) + 0xa5 (0x2aaaacb0fb65 in /lib/libtorch_cpu.so)
frame #4: <unknown function> + 0x1d11a26 (0x2aaaad268a26 in /lib/libtorch_cpu.so)
frame #5: at::_ops::_cat::call(c10::ArrayRef<at::Tensor>, long) + 0x17a (0x2aaaad12163a in /lib/libtorch_cpu.so)
frame #6: at::native::cat(c10::ArrayRef<at::Tensor>, long) + 0xc1 (0x2aaaacb1b271 in /lib/libtorch_cpu.so)
frame #7: <unknown function> + 0x1e3b976 (0x2aaaad392976 in /lib/libtorch_cpu.so)
frame #8: at::_ops::cat::redispatch(c10::DispatchKeySet, c10::ArrayRef<at::Tensor>, long) + 0xd9 (0x2aaaacdd4af9 in /lib/libtorch_cpu.so)
frame #9: <unknown function> + 0x2a40b40 (0x2aaaadf97b40 in /lib/libtorch_cpu.so)
frame #10: <unknown function> + 0x2a413c9 (0x2aaaadf983c9 in /lib/libtorch_cpu.so)
frame #11: at::_ops::cat::call(c10::ArrayRef<at::Tensor>, long) + 0x17a (0x2aaaace17fda in /lib/libtorch_cpu.so)
frame #12: /home/ubuntu/haskell/.stack-work/install/x86_64-linux-tinfo6-libc6-pre232/35051025c4a963044e43315a45cb6d083bfb428ddd62214b32228fdf98896950/9.2.8/bin/session7-rnn() [0x5cfb26]
frame #13: /home/ubuntu/haskell/.stack-work/install/x86_64-linux-tinfo6-libc6-pre232/35051025c4a963044e43315a45cb6d083bfb428ddd62214b32228fdf98896950/9.2.8/bin/session7-rnn() [0x5c0f3d]
; type: c10::Error
session7-rnn: CppStdException e "Tensors must have same number of dimensions: got 2 and 1\nException raised from check_cat_shape_except_dim at ../aten/src/ATen/native/TensorShape.h:12 (most recent call first):\nframe #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x6b (0x2aaaab50c7ab in /lib/libc10.so)\nframe #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0xce (0x2aaaab50815e in /lib/libc10.so)\nframe #2: at::native::_cat_out_cpu(c10::ArrayRef<at::Tensor>, long, at::Tensor&) + 0xae6 (0x2aaaacb0f596 in /lib/libtorch_cpu.so)\nframe #3: at::native::_cat_cpu(c10::ArrayRef<at::Tensor>, long) + 0xa5 (0x2aaaacb0fb65 in /lib/libtorch_cpu.so)\nframe #4: <unknown function> + 0x1d11a26 (0x2aaaad268a26 in /lib/libtorch_cpu.so)\nframe #5: at::_ops::_cat::call(c10::ArrayRef<at::Tensor>, long) + 0x17a (0x2aaaad12163a in /lib/libtorch_cpu.so)\nframe #6: at::native::cat(c10::ArrayRef<at::Tensor>, long) + 0xc1 (0x2aaaacb1b271 in /lib/libtorch_cpu.so)\nframe #7: <unknown function> + 0x1e3b976 (0x2aaaad392976 in /lib/libtorch_cpu.so)\nframe #8: at::_ops::cat::redispatch(c10::DispatchKeySet, c10::ArrayRef<at::Tensor>, long) + 0xd9 (0x2aaaacdd4af9 in /lib/libtorch_cpu.so)\nframe #9: <unknown function> + 0x2a40b40 (0x2aaaadf97b40 in /lib/libtorch_cpu.so)\nframe #10: <unknown function> + 0x2a413c9 (0x2aaaadf983c9 in /lib/libtorch_cpu.so)\nframe #11: at::_ops::cat::call(c10::ArrayRef<at::Tensor>, long) + 0x17a (0x2aaaace17fda in /lib/libtorch_cpu.so)\nframe #12: /home/ubuntu/haskell/.stack-work/install/x86_64-linux-tinfo6-libc6-pre232/35051025c4a963044e43315a45cb6d083bfb428ddd62214b32228fdf98896950/9.2.8/bin/session7-rnn() [0x5cfb26]\nframe #13: /home/ubuntu/haskell/.stack-work/install/x86_64-linux-tinfo6-libc6-pre232/35051025c4a963044e43315a45cb6d083bfb428ddd62214b32228fdf98896950/9.2.8/bin/session7-rnn() [0x5c0f3d]\n"(Just "c10::Error")
```
