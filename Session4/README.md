stack run session4-perseptron-and-gate
stack run session4-mlpxor
stack run session4-step-mlpxor

## A Simple Perceptron
### My results
learning rate = 0.1  
epoch = 150  
$ stack run session4-perseptron-and-gate  
initial weights [0.48967326,0.21364456]  
initial bias 0.97732216  
initial error[0.0,-1.0,-1.0,-1.0]  
final weights [0.18967327,0.21364456]  
final bias -0.22267789  
final error[0.0,0.0,0.0,0.0]    



## A Multi-layer Perceptron
### 2.b My explanation about the code

Define the type of MLPSpec and MLP.
```haskell:MlpXor.hs
data MLPSpec = MLPSpec
  { feature_counts :: [Int],   -- 層のサイズ
    nonlinearitySpec :: Tensor -> Tensor  -- 活性化関数
  }

data MLP = MLP
  { layers :: [Linear],   -- 線形層のリスト
    nonlinearity :: Tensor -> Tensor   -- 活性化関数
  }
  deriving (Generic, Parameterized)
```

MLPSpec and MLP correspond to spec and f of the typeclass Randomizable, respectively.  
Sample is a function that randomly generates MLP based on information of MLPSpec.
```haskell:MlpXor.hs
-- How to make random MLP from MLPSpec
instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) =
        scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)
```

Apply functions of every layer to input. Functions to be applied to input are in the list, specifically in the following form.  
[Σ(w*x+b), nonlinearity, Σ(w*x+b) ..]
```haskell:MlpXor.hs
mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x
```

batchSize is the size of a dataset when it is divided into several groups.  
numIters is the number of iterations.
```haskell:MlpXor.hs
batchSize = 2

numIters = 2000

model :: MLP -> Tensor -> Tensor
model params t = mlp params t
```

Initialize the MLP by specifying the layer size and activation function.
```haskell:MlpXor.hs
main :: IO ()
main = do
  -- generate MLP
  init <-
    sample $
      MLPSpec
        { feature_counts = [2, 2, 1],
          nonlinearitySpec = Torch.tanh
        }
```

Update training data, calculate MSE, and update weights and bias at each iteration of learning.
```haskell:MlpXor.hs
  trained <- foldLoop init numIters $ \state i -> do
    -- generate learning data
    input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5)
    -- calculate the MSE
    let (y, y') = (tensorXOR input, squeezeAll $ model state input)
        loss = mseLoss y y'
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    -- update weights and bias
    (newState, _) <- runStep state optimizer loss 1e-1
    return newState
```

Output the results and function to compute the correct output (y)
```haskell:MlpXor.hs
  putStrLn "Final Model:"
  putStrLn $ "0, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 0 :: Float]))
  putStrLn $ "0, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 1 :: Float]))
  putStrLn $ "1, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 0 :: Float]))
  putStrLn $ "1, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 1 :: Float]))
  return ()
  where
    optimizer = GD
    tensorXOR :: Tensor -> Tensor
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = select 1 0 t
        b = select 1 1 t
```

Here is the definition of the function I referred to.
```haskell:NN.hs
type Parameter = IndependentTensor

class Randomizable spec f | spec -> f where
  sample :: spec -> IO f

data Linear = Linear
  { weight :: Parameter,
    bias :: Parameter
  }
  deriving (Show, Generic, Parameterized)

data LinearSpec = LinearSpec
  { in_features :: Int,
    out_features :: Int
  }
  deriving (Show, Eq)

instance Randomizable LinearSpec Linear where
  sample LinearSpec {..} = do ..

linear :: Linear -> Tensor -> Tensor
```


```haskell:Functional.hs
squeezeAll ::
  -- | input
  Tensor ->
  -- | output
  Tensor
squeezeAll t = unsafePerformIO $ cast1 ATen.squeeze_t t

tanh ::
  -- | input
  Tensor ->
  -- | output
  Tensor
tanh t = unsafePerformIO $ cast1 ATen.tanh_t t

mseLoss ::
  -- | target tensor
  Tensor ->
  -- | input
  Tensor ->
  -- | output
  Tensor
mseLoss target t = unsafePerformIO $ cast3 ATen.mse_loss_ttl t target ATen.kMean
```

```haskell:Optim.hs
runStep ::
  forall model optim parameters gradients tensors dtype device.
  ( Parameterized model,
    parameters ~ Parameters model,
    HasGrad (HList parameters) (HList gradients),
    tensors ~ gradients,
    HMap' ToDependent parameters tensors,
    ATen.Castable (HList gradients) [D.ATenTensor],
    Optimizer optim gradients tensors dtype device,
    HMapM' IO MakeIndependent tensors parameters
  ) =>
  model ->
  optim ->
  Loss device dtype ->
  LearningRate device dtype ->
  IO (model, optim)
```