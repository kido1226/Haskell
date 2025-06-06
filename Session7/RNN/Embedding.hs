{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Embedding where

import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch

data EmbeddingSpec = EmbeddingSpec
  { wordNum :: Int, -- the number of words
    wordDim :: Int -- the dimention of word embeddings
  }
  deriving (Show, Eq, Generic)

data Embedding = Embedding
  { wordEmbedding :: Parameter -- Parameter = IndependentTensor
  }
  deriving (Show, Generic, Parameterized)

instance Randomizable EmbeddingSpec Embedding where
  sample EmbeddingSpec {..} = do
    -- ランダムな埋め込み行列（標準正規分布）を生成: <wordNum, wordDim>
    embedTensor <- randnIO' [wordNum, wordDim]
    -- パラメータ化して返す
    wordEmbedding <- makeIndependent embedTensor
    return $ Embedding wordEmbedding