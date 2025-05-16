{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch
import ML.Exp.Chart   (drawLearningCurve)

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
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize = 2

numIters = 2000

model :: MLP -> Tensor -> Tensor
model params t = mlp params t

step' :: Tensor -> Tensor
step' x = threshold 0.0 1.0 (0.0 - (threshold 0.0 0.0 x))

main :: IO ()
main = do
  -- generate MLP
  init <-
    sample $
      MLPSpec
        { feature_counts = [2, 2, 1],
          nonlinearitySpec = step'
        }

  -- foldLoop :: a -> Int -> (a -> Int -> IO a) -> IO a
  (trained, losses) <- foldLoop (init, []) numIters $ \(state,losses) i -> do
    -- generate learning data
    input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5)
    -- calculate the MSE
    let (y, y') = (tensorXOR input, squeezeAll $ model state input)
        loss = mseLoss y y'
        loss' = (asValue loss :: Float)
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    -- update weights and bias
    (newState, _) <- runStep state optimizer loss 1e-1
    return (newState, (loss' : losses))

  drawLearningCurve "stepMlpXor/step-xor.png" "Learning Curve" [("",reverse losses)]
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