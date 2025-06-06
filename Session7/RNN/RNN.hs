{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

import Codec.Binary.UTF8.String (encode)
import Control.Monad (forM_, when)
import Data.Aeson (FromJSON (..), ToJSON (..), eitherDecode)
import Data.ByteString.Char8 qualified as BC
import Data.ByteString.Lazy qualified as B
import Data.Char (isAlphaNum)
import Data.List (nub)
import Data.Map.Strict qualified as M
import Data.Word (Word8)
-- Hasktorch core

import Embedding
import GHC.Generics
import JsonlParser
import ML.Exp.Chart (drawLearningCurve)
import MLP
import Torch.Autograd (makeIndependent, toDependent)
import Torch.DType
import Torch.Device qualified as D
import Torch.Functional (Dim (..), KeepDim (..), embedding', logSoftmax, matmul, meanDim, mseLoss, tanh, transpose2D, unsqueeze)
import Torch.Layer.NonLinear (ActName (..))
import Torch.Layer.RNN as RNN
import Torch.NN (Linear, Parameter, Parameterized (..), Randomizable (..))
import Torch.Optim (GD (..), Optimizer, foldLoop, mkAdam, runStep)
import Torch.Serialize (loadParams, saveParams)
import Torch.Tensor (Tensor, asTensor, asValue, shape)
import Torch.TensorFactories (eye', randIO', randnIO', zeros', zerosLike)

-- model
data ModelSpec = ModelSpec
  { embSpec :: EmbeddingSpec,
    rnnHypParams :: RnnHypParams,
    h0Params :: InitialStatesHypParams,
    mlpSpec :: MLPSpec
  }
  deriving (Generic)

data Model = Model
  { emb :: Embedding,
    rnn :: RnnParams,
    initialStates :: InitialStatesParams,
    mlp :: MLP
  }
  deriving (Generic, Parameterized)

-- randomize and initialize embedding with loaded params
initialize ::
  ModelSpec ->
  FilePath ->
  IO Model
initialize modelSpec@ModelSpec {..} embPath = do
  -- embはrandEmbかloadedembか選ぶ
  -- 埋め込みのランダム初期化
  randEmb <- sample embSpec
  -- 埋め込みパラメータをロード
  loadedEmb <- loadParams randEmb embPath
  -- 以下rnn,h0,mlpの初期化
  rnn <- sample rnnHypParams
  h0 <- sample h0Params
  mlp <- sample mlpSpec
  return $ Model {emb = loadedEmb, rnn = rnn, initialStates = h0, mlp = mlp}

isUnncessaryChar :: Char -> Bool
isUnncessaryChar c = not (isAlphaNum c || c == ' ')

preprocess :: String -> [B.ByteString]
preprocess str =
  let cleaned = filter (not . isUnncessaryChar) str
   in map (B.fromStrict . BC.pack) (words cleaned)

wordToIndexFactory ::
  [B.ByteString] -> -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0 .. length wordlst]))

padToLength :: Int -> Int -> [[Int]] -> [[Int]]
padToLength maxLen padId xs = map pad xs
  where
    pad ys
      | length ys < maxLen = ys ++ replicate (maxLen - length ys) padId
      | otherwise = take maxLen ys

-- your amazon review json
amazonReviewPath :: FilePath
amazonReviewPath = "Session7/data/sample.jsonl"

outputPath :: FilePath
outputPath = "Session7/data/review-texts.txt"

embeddingPath = "Session6/data/sample_embedding.params"

wordLstPath = "Session6/data/sample_wordlst.txt"

numIters = 100

rate = 0.01

main :: IO ()
main = do
  jsonl <- B.readFile amazonReviewPath
  let amazonReviews = decodeToAmazonReview jsonl
  let reviews = case amazonReviews of
        Left err -> []
        Right reviews -> reviews

  -- text(input)とrating(output)を取り出す
  let textLines = map (preprocess . text) reviews
  let ratings = map (\x -> x - 1) (map rating reviews)
  let ratings' = asTensor (ratings :: [Int])

  -- load word list (It's important to use the same list as whan creating embeddings)
  wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)

  let wordToIndex = wordToIndexFactory wordLst
  let indexedData = map (map wordToIndex) textLines
  let texts = padToLength 20 0 indexedData
  let inputTensor = asTensor (texts :: [[Int]])
  print $ shape ratings'
  print $ shape inputTensor

  -- load params (set　wordDim　and wordNum same as session5)
  let embSpec =
        EmbeddingSpec
          { wordDim = 9,
            wordNum = length wordLst + 1
          }
  let rnnHypParams =
        RnnHypParams
          { inputSize = wordDim embSpec,
            hiddenSize = 128,
            numLayers = 1,
            bidirectional = False,
            hasBias = True,
            dev = D.Device D.CPU 0
          }
  -- rnnHypParamsと同じにする
  let h0Params =
        InitialStatesHypParams
          { dev = D.Device D.CPU 0,
            bidirectional = False,
            hiddenSize = 128,
            numLayers = 1
          }
  let mlpSpec =
        MLPSpec
          { feature_counts = [128, 20, 5],
            nonlinearitySpec = logSoftmax (Dim 1)
          }
  let modelSpec =
        ModelSpec
          { embSpec = embSpec,
            rnnHypParams = rnnHypParams,
            h0Params = h0Params,
            mlpSpec = mlpSpec
          }
  initModel <- initialize modelSpec embeddingPath

  let optimizer = GD

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
  -- 学習曲線を描画
  drawLearningCurve "Session7/RNN" "Learning Curve" [("", reverse losses)]

  return ()
