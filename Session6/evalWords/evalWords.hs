{-# LANGUAGE OverloadedStrings #-}

import Codec.Binary.UTF8.String (encode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as BC
import Data.Char (isDigit)
import Data.Maybe (mapMaybe)
import Data.Word (Word8)
import Text.Read (readMaybe)

testFilePath = "data/answer_test.tsv"

-- 空白文字で分割して単語のByteStringリストを返す関数（記号除去付き）
tokenizeSentence :: BC.ByteString -> [BC.ByteString]
tokenizeSentence =
  filter (not . isUnnecessary) . BC.split ' '
  where
    isUnnecessary word = word `elem` [".", "!", "?"]

-- ラベルをIntとして安全に読み取る
parseLabel :: B.ByteString -> Maybe Int
parseLabel label = readMaybe (BC.unpack label)

preprocess :: B.ByteString -> [(Int, [BC.ByteString], [BC.ByteString])]
preprocess texts =
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

main :: IO ()
main = do
  tests <- B.readFile testFilePath
  let testLines = preprocess tests
  print testLines
