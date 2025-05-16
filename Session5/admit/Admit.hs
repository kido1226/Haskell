{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}  -- toType
import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch
import ML.Exp.Chart (drawLearningCurve)
import Numeric (showFFloat)

import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V
import Data.Csv

import Evaluation

data Input = Input
  { greScore          :: !Float,
    toeflScore        :: !Float,
    universityRating  :: !Float,
    sop               :: !Float,
    lor               :: !Float,
    cgpa              :: !Float,
    research          :: !Float,
    chanceOfAdmit     :: !Float
  } deriving (Show, Generic)

-- csvのヘッダー名とInputのフィールド名が不一致の時はwhere以下必須
instance FromRecord Input

load' :: FilePath -> IO (Tensor,Tensor)
load' filePath = do
    csvData <- BL.readFile filePath
    case decode HasHeader csvData of               -- decodeByNameだとvalid.csvが読み込めない
        Left err -> error err
        Right v -> do                     -- v is vector of records
            let rows = V.toList v
                xsList = map (\row -> [ greScore row
                     , toeflScore row
                     , universityRating row
                     , sop row
                     , lor row
                     , cgpa row
                     , research row ])
                    rows
                ysList = map chanceOfAdmit rows
                xsTensor = asTensor xsList
                ysTensor = asTensor ysList
            return (xsTensor, ysTensor)



--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

data MLPSpec = MLPSpec
  { feature_counts :: [Int],   -- 層のサイズ
    nonlinearitySpec :: Tensor -> Tensor  -- 活性化関数の仕様
  }

data MLP = MLP
  { layers :: [Linear],   -- 線形層のリスト
    nonlinearity :: Tensor -> Tensor   -- 活性化関数本体
  }
  deriving (Generic, Parameterized)

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

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = sigmoid $ foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize = 2

numIters = 1000

rate = 0.01

model :: MLP -> Tensor -> Tensor
model params t = mlp params t


--出力用
-- ヘルパー関数：Floatを小数点以下4桁にフォーマット
format4 :: Float -> String
format4 x = showFFloat (Just 4) x ""

-- リスト出力用（Accuracy, Precision, Recall, F1）
printListMetric :: String -> [Float] -> IO ()
printListMetric name values = do
  putStrLn $ name ++ ": [" ++ formatted ++ "]"
  where
    formatted = unwords $ map (\x -> format4 x) values

-- 単一値出力用（macroF1, weightedF1, microF1）
printSingleMetric :: String -> Float -> IO ()
printSingleMetric name value = do
  putStrLn $ name ++ ": " ++ format4 value


main :: IO ()
main = do
  -- read csv
  (xs, ys) <- load' "admit/data/train.csv"
  let ys' = toType Float ys

  -- generate MLP
  init <-
    sample $
      MLPSpec
        { feature_counts = [7, 16, 8, 1],
          nonlinearitySpec = sigmoid
        }

  -- foldLoop :: a -> Int -> (a -> Int -> IO a) -> IO a
  (trained, losses) <- foldLoop (init, []) numIters $ \(state,losses) i -> do
    -- generate learning data
    let input = xs
    -- calculate the MSE
    let (y, y') = (ys', squeezeAll $ model state input)
        loss = mseLoss y y'
        loss' = (asValue loss :: Float)
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    -- update weights and bias
    (newState, _) <- runStep state optimizer loss rate
    return (newState, (loss' : losses))

  drawLearningCurve "admit/admit.png" "Learning Curve" [("",reverse losses)]



  -- validで確認
  (vx, vy) <- load' "admit/data/valid.csv"

  let predY = ge (squeezeAll $ model trained vx) (asTensor(0.65 :: Float))
  let actualY = ge vy (asTensor(0.65 :: Float))
  
  putStrLn $ "Actual value:\n" ++ show vy
  putStrLn $ "Predicted value:\n" ++ show (squeezeAll $ model trained vx)
  putStrLn $ "Actual value:\n" ++ show actualY
  putStrLn $ "Predicted value:\n" ++ show predY


  -- evaluate the model
  let con = confusionMatrix 2 predY actualY
  let acc = accuracy con
  let prec = precision con
  let rcl = recall con
  let f1 = f1score con
  let macrof1 = macroF1 con
  let wf1 = weightedF1 con
  let microf1 = microF1 con

  print con
  printListMetric "Accuracy" acc
  printListMetric "Precision" prec
  printListMetric "Recall" rcl
  printListMetric "F1 Score" f1
  printSingleMetric "Macro F1 Score" macrof1
  printSingleMetric "Weighted F1 Score" wf1
  printSingleMetric "Micro F1 Score" microf1

  return ()
  where
    optimizer = GD