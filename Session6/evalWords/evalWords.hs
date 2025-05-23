{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import Codec.Binary.UTF8.String (encode)
import Control.Monad (mapM_)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as BC
import Data.Char (isDigit)
import Data.List (nub)
import qualified Data.Map.Strict as M
import Data.Maybe (mapMaybe)
import Data.Word (Word8)
import GHC.Generics
import Text.Read (readMaybe)
import Torch.Autograd (makeIndependent, toDependent)
import Torch.DType
import Torch.Functional (Dim (..), KeepDim (..), dot, embedding', meanDim, sqrt)
import Torch.NN (Parameter, Parameterized (..))
import Torch.Serialize (loadParams)
import Torch.Tensor (Tensor, asTensor, asValue, reshape, shape)
import Torch.TensorFactories (zeros')

testFilePath = "data/answer_test.tsv"

textFilePath = "data/sample.txt"

modelPath = "data/sample_embedding.params"

data Embedding = Embedding
  { wordEmbedding :: Parameter -- Parameter = IndependentTensor
  }
  deriving (Show, Generic, Parameterized)

wordToIndexFactory ::
  [B.ByteString] -> -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault 0 wrd (M.fromList (zip wordlst [0 .. length wordlst - 1]))

-- 空白文字で分割して単語のByteStringリストを返す関数（記号除去付き）
tokenizeSentence :: BC.ByteString -> [BC.ByteString]
tokenizeSentence =
  filter (not . isUnnecessary) . BC.split ' '
  where
    isUnnecessary word = word `elem` [".", "!", "?"]

-- ラベルをIntとして安全に読み取る
parseLabel :: B.ByteString -> Maybe Int
parseLabel label = readMaybe (BC.unpack label)

preprocess' :: B.ByteString -> [(Int, [BC.ByteString], [BC.ByteString])]
preprocess' texts =
  mapMaybe parseLine textLines
  where
    textLines = B.split (head $ encode "\n") texts

    parseLine :: B.ByteString -> Maybe (Int, [BC.ByteString], [BC.ByteString])
    parseLine line =
      case B.split (head $ encode "\t") line of
        (label : sent1 : sent2 : _) ->
          case parseLabel label of
            Just n -> Just (n, tokenizeSentence sent1, tokenizeSentence sent2)
            Nothing -> Nothing
        _ -> Nothing

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

-- tokensをemb化して平均値をとる
toAvgEmbedding :: [B.ByteString] -> (B.ByteString -> Int) -> Tensor -> Tensor
toAvgEmbedding tokens wordToIndex embMatrix =
  let idxs = map wordToIndex tokens
      embVecs = embedding' embMatrix (asTensor [idxs])
      meanVec = meanDim (Dim 1) RemoveDim Float embVecs
   in reshape [-1] meanVec -- <- 1D

norm :: Tensor -> Tensor
norm v = Torch.Functional.sqrt (v `dot` v)

cosineSim :: Tensor -> Tensor -> Float
cosineSim v1 v2 =
  let sim = (v1 `dot` v2) / (norm v1 * norm v2)
   in asValue sim

mapCosineToScore :: Float -> Int
mapCosineToScore sim
  | sim < -0.6 = 0
  | sim < -0.2 = 1
  | sim < 0.2 = 2
  | sim < 0.6 = 3
  | sim < 0.8 = 4
  | otherwise = 5

main :: IO ()
main = do
  tests <- B.readFile testFilePath
  let testLines = preprocess' tests
  print testLines

  texts <- B.readFile textFilePath
  -- Create a unique word list
  let wordLines = preprocess texts
      wordlst = nub $ concat wordLines
      wordToIndex = wordToIndexFactory wordlst

  -- reading embedding
  initWordEmb <- makeIndependent $ zeros' [1]
  let initEmb = Embedding {wordEmbedding = initWordEmb}
  loadedEmb <- loadParams initEmb modelPath
  let embMatrix = toDependent $ wordEmbedding loadedEmb

  mapM_
    ( \(val, s1, s2) -> do
        let s1emb = toAvgEmbedding s1 wordToIndex embMatrix
            s2emb = toAvgEmbedding s2 wordToIndex embMatrix
            evalCossim = cosineSim s1emb s2emb
            score = mapCosineToScore evalCossim
        putStrLn $ "actual similarity is " ++ show val
        putStrLn $ "estimated similarity is " ++ show score
    )
    testLines
