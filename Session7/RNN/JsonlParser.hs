{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE OverloadedStrings #-}

-- docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/hasktorch-nlp-introduction && stack run day6-parse"

module JsonlParser where

-- json
import Data.Aeson
import Data.ByteString.Internal qualified as B (c2w)
import Data.ByteString.Lazy qualified as B
import GHC.Generics

data Image = Image
  { small_image_url :: String,
    medium_image_url :: String,
    large_image_url :: String
  }
  deriving (Show, Generic)

instance FromJSON Image

instance ToJSON Image

data AmazonReview = AmazonReview
  { rating :: Int,
    title :: String,
    text :: String,
    images :: [Image],
    asin :: String,
    parent_asin :: String,
    user_id :: String,
    timestamp :: Int,
    verified_purchase :: Bool,
    helpful_vote :: Int
  }
  deriving (Show, Generic)

instance FromJSON AmazonReview

instance ToJSON AmazonReview

decodeToAmazonReview ::
  B.ByteString ->
  Either String [AmazonReview]
decodeToAmazonReview jsonl =
  let jsonList = B.split (B.c2w '\n') jsonl
   in sequenceA $ map eitherDecode jsonList