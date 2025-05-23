{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE StandaloneDeriving #-}

module Main (main) where

import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml

-- add bytestring to dependencies in package.yaml

-- add containers to dependencies in package.yaml

import Control.Monad (forM_, when)
import qualified Data.ByteString.Lazy as B
import Data.List (nub)
import qualified Data.Map.Strict as M
import Data.Word (Word8)
import GHC.Generics
import ML.Exp.Chart (drawLearningCurve)
import System.Random.Shuffle (shuffleM)
import Torch.Autograd (makeIndependent, toDependent)
import Torch.DType
import Torch.Functional (Dim (..), KeepDim (..), embedding', logSoftmax, matmul, meanDim, nllLoss', transpose2D)
import Torch.NN
import Torch.NN (Linear, Parameter, Parameterized (..), Randomizable (..))
import Torch.Optim (GD (..), foldLoop, runStep)
import Torch.Serialize (loadParams, saveParams)
import Torch.Tensor (Tensor, asTensor, asValue, shape)
import Torch.TensorFactories (eye', randIO', zeros')

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

data MLPSpec = MLPSpec
  { feature_counts :: [Int], -- 層のサイズ
    nonlinearitySpec :: Tensor -> Tensor -- 活性化関数の仕様
  }

data MLP = MLP
  { layers :: [Linear], -- 線形層のリスト
    nonlinearity :: Tensor -> Tensor -- 活性化関数本体
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

-- your text data (try small data first)
textFilePath = "data/sample.txt"

modelPath = "data/sample_embedding.params"

wordLstPath = "data/sample_wordlst.txt"

data EmbeddingSpec = EmbeddingSpec
  { wordNum :: Int, -- the number of words
    wordDim :: Int -- the dimention of word embeddings
  }
  deriving (Show, Eq, Generic)

data Embedding = Embedding
  { wordEmbedding :: Parameter -- Parameter = IndependentTensor
  }
  deriving (Show, Generic, Parameterized)

-- Probably you should include model and Embedding in the same data class.
data Model = Model
  { mlp :: MLP,
    embeddings :: Embedding
  }
  deriving (Generic, Parameterized)

isUnncessaryChar ::
  Word8 ->
  Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "!"]

preprocess ::
  B.ByteString -> -- input
  [[B.ByteString]] -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory ::
  [B.ByteString] -> -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0 .. length wordlst]))

toyEmbedding ::
  EmbeddingSpec ->
  Tensor -- embedding
toyEmbedding EmbeddingSpec {..} =
  eye' wordNum wordDim

-- 各リストに対してバッチ処理を適用する
makeBatches ::
  Int -> -- window size
  [[Int]] -> -- data
  [([Int], Int)]
makeBatches windowSize wordData =
  concatMap (makeBatchesForList windowSize) wordData
  where
    makeBatchesForList :: Int -> [Int] -> [([Int], Int)]
    makeBatchesForList n xs
      | length xs < 2 * n + 1 = []
      | otherwise =
          [ (Prelude.take n (drop (i - n) xs) ++ Prelude.take n (drop (i + 1) xs), xs !! i)
            | i <- [n .. length xs - n - 1]
          ]

-- ミニバッチ学習用の関数
build :: ((a -> [a] -> [a]) -> [a] -> [a]) -> [a]
build g = g (:) []

chunks :: Int -> [e] -> [[e]]
chunks i ls = map (take i) (build (splitter ls))
  where
    splitter :: [e] -> ([e] -> a -> a) -> a -> a
    splitter [] _ n = n
    splitter l c n = l `c` splitter (drop i l) c n

numIters = 100

rate = 0.1

main :: IO ()
main = do
  -- load text file
  texts <- B.readFile textFilePath

  -- Create a unique word list
  let wordLines = preprocess texts
      wordlst = nub $ concat wordLines
      wordToIndex = wordToIndexFactory wordlst
      indexedData = map (map wordToIndex) wordLines
  print wordlst

  -- Create initial embedding (wordDim × wordNum)
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst + 1, wordDim = 9}
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
  let emb = Embedding {wordEmbedding = wordEmb}

  let batches = makeBatches 1 indexedData
  print $ length batches

  --   -- TODO: Train model. After training, we can obtain the trained patameter, embeddings. This is the trained embedding.
  --   let (contexts, targets) = unzip batches
  --       contextsTensor = asTensor (contexts :: [[Int]])
  --       targetsTensor = asTensor (targets :: [Int])
  --   -- foldLoop :: a -> Int -> (a -> Int -> IO a) -> IO a
  --   (trainedEmb, losses) <- foldLoop (emb, []) numIters $ \(state, losses) i -> do
  --     let contextVecs = embedding' (toDependent $ wordEmbedding state) contextsTensor -- [batch_size, context_size, emb_dim]
  --         avgVecs = meanDim (Dim 1) RemoveDim Float contextVecs -- [batch_size, embed_dim]
  --         scores = matmul avgVecs (transpose2D (toDependent $ wordEmbedding state)) -- [batch_size, vocab_size]
  --         probs = logSoftmax (Dim 1) scores
  --         loss = nllLoss' targetsTensor probs
  --     print loss
  --     let loss' = (asValue loss :: Float)
  --     when (i `mod` 10 == 0) $ do
  --       putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
  --     (newState, _) <- runStep state optimizer loss rate
  --     return (newState, (loss' : losses))

  shuffledBatches <- shuffleM batches
  let miniBatches = chunks 100 shuffledBatches
  (trainedEmb, allLosses) <- foldLoop (emb, []) numIters $ \(state, losses) i -> do
    (state', epochLosses) <- foldLoop (state, []) ((length miniBatches) - 1) $ \(s, ls) batchIdx -> do
      let miniBatch = miniBatches !! batchIdx
          (contexts, targets) = unzip miniBatch
          contextsTensor = asTensor (contexts :: [[Int]])
          targetsTensor = asTensor (targets :: [Int])

          contextVecs = embedding' (toDependent $ wordEmbedding s) contextsTensor
          avgVecs = meanDim (Dim 1) RemoveDim Float contextVecs
          scores = matmul avgVecs (transpose2D (toDependent $ wordEmbedding s))
          probs = logSoftmax (Dim 1) scores
          loss = nllLoss' targetsTensor probs
          loss' = asValue loss :: Float
      when (i `mod` 10 == 0 && batchIdx `mod` 100 == 0) $
        putStrLn $
          "Epoch: " ++ show i ++ ", Batch: " ++ show batchIdx ++ " | Loss: " ++ show loss'

      (newState, _) <- runStep s optimizer loss rate
      return (newState, loss' : ls)

    let meanLoss = (sum epochLosses) / fromIntegral (length epochLosses)
    return (state', meanLoss : losses)

  drawLearningCurve "word2vec/word2vec_b100.png" "Learning Curve" [("", reverse allLosses)]

  -- Save params to use trained parameter in the next session
  -- trainedEmb :: Embedding
  saveParams trainedEmb modelPath
  -- Save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)

  -- Load params
  initWordEmb <- makeIndependent $ zeros' [1]
  let initEmb = Embedding {wordEmbedding = initWordEmb}
  loadedEmb <- loadParams initEmb modelPath

  let sampleTxt = B.pack $ encode "This is awesome.\nmodel is developing"
  -- convert word to index
  let idxes = map (map wordToIndex) (preprocess sampleTxt)
  -- convert to embedding
  let embTxt = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor idxes)
  print idxes
  print embTxt

  return ()
  where
    optimizer = GD