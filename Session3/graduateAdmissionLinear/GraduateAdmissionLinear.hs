{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import Control.Monad (zipWithM_)
import ML.Exp.Chart (drawLearningCurve)
import Torch.Functional (add, mul, sumAll)
import Torch.Tensor (Tensor, asTensor, asValue, numel)

import GHC.Generics (Generic)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V
import Data.Csv

data Input = Row
  { x :: !Float
  , y :: !Float
  } deriving (Show, Eq, Generic)

-- csvのヘッダー名とInputのフィールド名が不一致の時はwhere以下必須
instance FromRecord Input


load :: FilePath -> IO (Tensor,Tensor)
load filePath = do
    csvData <- BL.readFile filePath
    case decode HasHeader csvData of               -- decodeByNameだとvalid.csvが読み込めない
        Left err -> error err
        Right v -> do                     -- v is vector of records
            let xsList = V.toList $ V.map x v  -- extract x values from v and convert to a list
                ysList = V.toList $ V.map y v  -- extract y values from v and convert to a list
                xsTensor = asTensor xsList
                ysTensor = asTensor ysList
            return (xsTensor, ysTensor)

rateA :: Tensor -- learning rate
rateA = asTensor (0.00005 :: Float)
rateB :: Tensor -- learning rate
rateB = asTensor (0.0012 :: Float)

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
      m = asTensor ((fromIntegral (numel z) :: Float) :: Float)
   in (sumAll squared) / m

calculateNewA ::
  Tensor ->   -- estimatedY
  Tensor ->   -- oldA
  Tensor ->   -- xs
  Tensor ->   -- ys
  Tensor   -- newA
calculateNewA estimatedY a xs ys =
  let diffs = estimatedY - ys
      m = asTensor ((fromIntegral (numel ys) :: Float) :: Float)
      grad = 2.0 * (sumAll $ (mul xs diffs)) / m
   in a - grad * rateA

calculateNewB ::
  Tensor ->   -- estimatedY
  Tensor ->   -- oldA
  Tensor ->   -- xs
  Tensor ->   -- ys
  Tensor   -- newA
calculateNewB estimatedY b xs ys =
  let diffs = estimatedY - ys
      m = asTensor ((fromIntegral (numel ys) :: Float) :: Float)
      grad = 2.0 * (sumAll diffs) / m
   in b - grad * rateB

train ::
  Int ->        -- epoch
  Tensor ->     -- initialA
  Tensor ->     -- initialB
  (Tensor, Tensor) ->     -- train.csv (xs, ys)
  (Tensor, Tensor) ->     -- valid.csv (vx, vy)
  [Float] ->    -- loss[]
  [Float] ->    -- lossv[]
  IO ((Tensor, Tensor), [Float], [Float])    -- ((finalA, finalB), losses)
train 0 a b (xs, ys) (vx, vy) loss lossv = return ((a, b), reverse loss, reverse lossv)
train n a b (xs, ys) (vx, vy) loss lossv = do
  let estimatedY = linear (a, b) xs
  let a' = calculateNewA estimatedY a xs ys 
  let b' = calculateNewB estimatedY b xs ys
  let cost' = cost ys (linear (a', b') xs)
  let costv = cost vy (linear (a', b') vx)
  putStrLn $ "Epoch " ++ show (epoch - n + 1) ++ ": cost = " ++ show (asValue cost' :: Float) ++ ": a = " ++ show (asValue a' :: Float) ++ ": b = " ++ show (asValue b' :: Float)
  putStrLn $ "valid cost " ++ show (asValue costv :: Float)
  train (n - 1) a' b' (xs, ys) (vx, vy) ((asValue cost' :: Float) : loss) ((asValue costv :: Float) : lossv)

main :: IO ()
main = do
  (xs, ys) <- load "graduateAdmissionLinear/data/train.csv"


  let initialA = asTensor (1.0 :: Float)
  let initialB = asTensor (1.0 :: Float)

  (vx, vy) <- load "graduateAdmissionLinear/data/valid.csv"
  (trainedData, costs, costv) <- train epoch initialA initialB (xs, ys) (vx, vy) [] []

  print trainedData

  let ys' = (asValue ys :: [Float])
  let estimatedY = linear trainedData xs
  let estimatedY' = (asValue estimatedY :: [Float])

--   zipWithM_
--     ( \a b -> do
--         putStrLn $ "correct answer: " ++ show b
--         putStrLn $ "estimated: " ++ show a
--         putStrLn "******"
--     )
--     estimatedY'
--     ys'

  drawLearningCurve "graduateAdmissionLinear/LearningCurve.png" "LearningCurve" [("LearningLoss", costs)]
  drawLearningCurve "graduateAdmissionLinear/ValidLearningCurve.png" "LearningCurve" [("ValidLearningLoss", costv)]