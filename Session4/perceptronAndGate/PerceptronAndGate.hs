module Main where
import Torch.Functional (add, mul, sumAll)
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (randIO')
import Torch.Functional.Internal (ge)

trainingData :: [([Int],Int)]
trainingData = [([1,1],1),([1,0],0),([0,1],0),([0,0],0)]

-- To use map, this is a list. Tensor is not Functor class.(fmap×)
xs :: [Tensor]   -- 4×2
xs = map (\(x, _) -> asTensor (map fromIntegral x :: [Float])) trainingData

output :: [Tensor]
output = map (\x' -> asTensor (x' :: Float)) (map (fromIntegral . snd) trainingData)

rate :: Tensor
rate = asTensor(0.1 :: Float)

epoch :: Int
epoch = 150

step :: Tensor -> Tensor
step net =
    let zero = asTensor(0 :: Float)
    in if  (asValue (ge net zero) :: Bool) then asTensor(1 :: Float) else asTensor(0 :: Float)

perceptron ::
  Tensor -> -- x 2×1
  Tensor -> -- weights 2×1
  Tensor -> -- bias 1×1
  Tensor    -- output 1×1
perceptron x w bias = step(sumAll(w * x) + bias)

calculateError ::
  Tensor ->   -- weights 2×1
  Tensor ->   -- bias
  [Tensor]    -- 4×1
calculateError w b =
    let f_net = map (\x -> perceptron x w b) xs  -- f_net :: [Tensor]
     in zipWith (-) output f_net

train ::
  Int ->     -- epoch
  Tensor ->  -- weights
  Tensor ->  -- bias
  IO (Tensor, Tensor)     -- (final weights, bias)
train 0 w b = return (w,b)
train n w b = do
    let error = calculateError w b
    let w' = w + rate * (foldl1 add (zipWith mul error xs))
    let b' = b + rate * (foldl1 add error)
    -- putStrLn $ 
    --   "Epoch " ++ show (epoch - n + 1) ++ 
    --   ": error = " ++ show (map (\e -> asValue e :: Float) error) ++ 
    --   ": w = " ++ show (asValue w' :: [Float]) ++ 
    --   ": b = " ++ show (asValue b' :: Float)
    train (n-1) w' b'

main :: IO ()
main = do
  -- initialize parameters(weights and bias) with random values.
  -- randIO' is impure, randn' is pure.
  w <- randIO' [2]
  b <- randIO' [1]
  putStrLn $ "initial weights " ++ show(asValue w :: [Float])
  putStrLn $ "initial bias " ++ show(asValue b :: Float)
  putStrLn $ "initial error" ++ show(map (\x -> asValue x :: Float) (calculateError w b))

  (weights, bias) <- train epoch w b
  putStrLn $ "final weights " ++ show(asValue weights :: [Float])
  putStrLn $ "final bias " ++ show(asValue bias :: Float)
  putStrLn $ "final error" ++ show(map (\x -> asValue x :: Float) (calculateError weights bias))
 
  -- train!
  return ()