module Evaluation(
    accuracy, 
    precision, 
    recall, 
    confusionMatrix, 
    f1score,
    macroF1,
    weightedF1,
    microF1) where

{-# LANGUAGE TypeApplications #-}  -- toType
import Torch
import Control.Monad(foldM)

indexAddOne ::
    Tensor -> -- [[Int]]
    (Int, Int) -> -- index
    Tensor -- [[Int]]
indexAddOne mat (i, j) =
    let mat' = asValue mat :: [[Float]]
        ans = addAt (i, j) mat'
    in asTensor(ans :: [[Float]])
    where
        addAt :: (Int, Int) -> [[Float]] -> [[Float]]
        addAt (i,j) mat =
            [if r == i then [if c == j then col+1
                                        else col
                            |(c,col) <- zip [0..] row]
                       else row
            |(r,row) <- zip [0..] mat]

accuracy ::
    Tensor -> -- confusion matrix
    [Float]
accuracy con =
    let con' = asValue con :: [[Float]]
        calc i =
            let tp = con' !! i !! i
                fp = sum [con' !! j !! i | j <- [0..(length con') - 1], j /= i] -- j!=i
                fn = sum [con' !! i !! j | j <- [0..(length con') - 1], j /= i]
                tn = (sum [sum row | row <- con']) - tp - fp - fn
            in (tp + tn) / (tp + tn + fp + fn)
    in [calc i | i <- [0..(length con') - 1]]

precision ::
    Tensor -> -- confusion matrix
    [Float]
precision con =
    let con' = asValue con :: [[Float]]
        calc i =
            let tp = con' !! i !! i
                fp = sum [con' !! j !! i | j <- [0..(length con') - 1], j /= i]
            in if (tp + fp) == 0 then 0 else tp / (tp + fp)
    in [calc i | i <- [0..(length con') - 1]]

recall ::
    Tensor -> -- confusion matrix
    [Float] 
recall con =
    let con' = asValue con :: [[Float]]
        calc i =
            let tp = con' !! i !! i
                fn = sum [con' !! i !! j | j <- [0..(length con') - 1], j /= i]
            in if (tp + fn) == 0 then 0 else tp / (tp + fn)
    in [calc i | i <- [0..(length con') - 1]]

confusionMatrix ::
    Int ->  -- class
    Tensor -> -- predicted values
    Tensor -> -- actual values
    Tensor    -- confusion matrix shape:class * class
confusionMatrix c preds actual =
    let actual' = toType Int64 actual
        preds' = toType Int64 preds
        zeros = zeros'([c, c] :: [Int])
        pairs = zip (asValue actual' :: [Int]) (asValue preds' :: [Int])
    in foldl indexAddOne zeros pairs

f1score ::
    Tensor -> -- confusion matrix
    [Float]
f1score con =
    let prec = precision con
        rcl = recall con
    in zipWith calc prec rcl
    where
        calc :: Float -> Float -> Float
        calc p c = if (p + c) == 0 then 0 else (2 * p * c) / (p + c)

macroF1 ::
    Tensor -> -- confusion matrix
    Float
macroF1 con =
    let f1 = f1score con
    in (sum f1) / (fromIntegral $ length f1)

weightedF1 ::
    Tensor -> -- confusion matrix
    Float
weightedF1 con =
    let con' = asValue con :: [[Float]]
        supports = map sum con'   -- 各行の和,データセットに含まれる各クラスの数
        f1 = f1score con
    in (sum $ zipWith (*) f1 supports) / (sum supports)

microF1 ::
    Tensor -> -- confusion matrix
    Float
microF1 con =
    let con' = asValue con :: [[Float]]
        tp = sum [con' !! i !! i | i <- [0..(length con') - 1]]
        fp = sum [sum [con' !! j !! i | j <- [0..(length con') - 1], j /= i] | i <- [0..(length con') - 1]]
        fn = sum [sum [con' !! i !! j | j <- [0..(length con') - 1], j /= i] | i <- [0..(length con') - 1]]
    in if (tp + (fp + fn)/2) == 0 then 0 else tp / (tp + (fp + fn)/2)