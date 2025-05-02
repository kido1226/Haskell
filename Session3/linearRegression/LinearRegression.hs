import Control.Monad (zipWithM_)
import ML.Exp.Chart (drawLearningCurve)
import Torch.Functional (add, mul, sumAll)
import Torch.Tensor (Tensor, asTensor, asValue, numel)

ys :: Tensor
ys = asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])

xs :: Tensor
xs = asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])

m :: Tensor
m = asTensor ((fromIntegral (numel ys) :: Float) :: Float)

rate :: Tensor -- learning rate
rate = asTensor (0.00001 :: Float)
rateA = asTensor (0.00001 :: Float)
rateB = asTensor (0.1  :: Float)

epoch :: Int -- number of iterate
epoch = 250

linear ::
  -- | parameters ([a, b]: 1 × 2, c: scalar)
  (Tensor, Tensor) ->
  -- | data x: 1 × 10
  Tensor ->
  -- | z: 1 × 10
  Tensor
linear (slope, intercept) input = (mul slope input) `add` intercept

-- slope and input are scalar

cost ::
  -- | grund truth: 1 × 10
  Tensor ->
  -- | estimated values: 1 × 10
  Tensor ->
  -- | loss: scalar
  Tensor
cost z z' =
  let diffs = z' - z
      squared = diffs * diffs
   in (sumAll squared) / m

calculateNewA ::
  Tensor ->
  Tensor ->
  IO Tensor
calculateNewA estimatedY a = do
  let diffs = estimatedY - ys
  let grad = 2.0 * (sumAll $ (mul xs diffs)) / m
  let a' = a - grad * rateA
  putStrLn $ "estimatedY = " ++ show (asValue estimatedY :: [Float]) ++ "\n diffs =  " ++ show (asValue diffs :: [Float]) ++ "\n grad =  " ++ show (asValue grad :: Float) ++ "\n a' =  " ++ show (asValue a' :: Float)
  return a'

calculateNewB ::
  Tensor ->
  Tensor ->
  IO Tensor
calculateNewB estimatedY b = do
  let diffs = estimatedY - ys
  let grad = 2.0 * (sumAll diffs) / m
  let b' = b - grad * rateB
  putStrLn $ "estimatedY = " ++ show (asValue estimatedY :: [Float]) ++ "\n diffs =  " ++ show (asValue diffs :: [Float]) ++ "\n grad =  " ++ show (asValue grad :: Float) ++ "\n b' =  " ++ show (asValue b' :: Float)
  return b'

train :: Int -> Tensor -> Tensor -> [Float] -> IO ((Tensor, Tensor), [Float])
train 0 a b loss = return ((a, b), reverse loss)
train n a b loss = do
  let estimatedY = linear (a, b) xs
  a' <- calculateNewA estimatedY a
  b' <- calculateNewB estimatedY b
  let cost' = cost ys (linear (a', b') xs)
  putStrLn $ "Epoch " ++ show (epoch - n + 1) ++ ": cost = " ++ show (asValue cost' :: Float) ++ ": a = " ++ show (asValue a' :: Float) ++ ": b = " ++ show (asValue b' :: Float)
  putStrLn "************************************************************"
  train (n - 1) a' b' ((asValue cost' :: Float) : loss)

-- if (asValue (cost ys (linear (a', b') xs)) :: Float) < 0.001
--    then (a', b')
--    else train (a', b')

main :: IO ()
main =
  do
    -- Below are pseudo code
    let sampleA = asTensor (0.1 :: Float)
    let sampleB = asTensor (0.1 :: Float)

    -- Iterate through the provided xs and ys data.
    (trainedData, costs) <- train epoch sampleA sampleB []
    print trainedData
    print costs

    -- -- For each pair, convert x to a tensor, calculate the estimatedY using your linear function with the provided sampleA and sampleB, and print both the correct y and the estimatedY.

    let estimatedY = linear trainedData xs

    let ys' = (asValue ys :: [Float])
    let estimatedY' = (asValue estimatedY :: [Float])

    zipWithM_
      ( \a b -> do
          putStrLn $ "correct answer: " ++ show b
          putStrLn $ "estimated: " ++ show a
          putStrLn "******"
      )
      estimatedY'
      ys'

    drawLearningCurve "linearRegression/LearningCurve.png" "LearningCurve" [("LearningLoss", costs)]

-- Expected outputs:
-- correct answer: 148
-- estimated: ?

-- correct answer: 186
-- ...