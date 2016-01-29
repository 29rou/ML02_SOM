import System.IO
import System.Environment (getArgs)
import Data.List as L
import Data.Vector.Storable as V
import Data.Int(Int8)
import Data.Word(Word8)
import System.Random(getStdRandom,randomR,randomRIO)
import Codec.Picture
import Codec.Picture.Types
import Graphics.Gloss(Picture(..),Display(..),white)
import Graphics.Gloss.Juicy(fromImageRGB8)
import Graphics.Gloss.Interface.IO.Simulate(simulateIO,ViewPort)

vExtract:: Vector Int -> Vector Int -> Int -> Int -> (Vector Int, Vector Int)
vExtract img ipixel n z = let n' = n*3
                              (z',indexL) = (z-1,fromList [(n'),(n')+1,(n')+2])
                              rgb =V.map (\x -> (img!x)*z') indexL
                              rgb' = V.map (\x -> div x z) $ V.zipWith (+) rgb ipixel
                          in (indexL, rgb')

vUpdate :: Vector Int -> Vector Int -> Int ->Vector Int
vUpdate img pixel index = let index' = index*3
                              lis = fromList [index',(index')+1,(index')+2]
                              pixeld = V.map (img!) lis
                              pixel' = V.map (\x -> div x 5)$ V.zipWith (+) pixeld (V.map (*4) pixel)
                              mod' m = ((mod index 50)/= m)
                              (vExtract',empty') = (vExtract img pixel,(empty,empty))
                              if' cod ind x  = if cod then vExtract' ind x else empty'
                              (iup,up)= if' (index>=50) (index-50) 5
                              (iup',up')= if' (index>=100) (index-100) 11
                              (idown,down)= if' (index<=2449) (index+50) 5
                              (idown',down')= if' (index<=2399) (index+100) 11
                              (iright,right)= if' (mod' 49) (index+1) 5
                              (iright',right')= if' ((mod' 49)&&(mod' 48)) (index+2) 11
                              (ileft,left)= if' (mod' 0) (index-1) 5
                              (ileft',left')= if' ((mod' 0)&&(mod' 1)) (index-2) 11
                              (iupright,upright)= if' (index>=50 && mod' 49) (index-49) 10
                              (iupleft,upleft)= if' (index>=50 && mod' 0) (index-51) 10
                              (idownright,downright)= if' (index<=2449 && mod' 49) (index+51) 10
                              (idownleft,downleft)= if' (index<=2449 && mod' 0) (index+49) 10
                              change = pixel' V.++ up V.++ up' V.++ down V.++ down' V.++ right V.++  right' V.++ left V.++ left' V.++ upright V.++ upleft V.++ downright V.++ downleft
                              target = lis V.++ iup V.++ iup' V.++ idown V.++ idown' V.++ iright V.++  iright' V.++ ileft V.++ ileft' V.++ iupright V.++ iupleft V.++ idownright V.++ idownleft
                          in update_ img target (change)

mainloop':: Vector Int -> IO (Vector Int)
mainloop' img = do let (img', file) = V.splitAt (7500) img
                   findex <- (randomRIO (0,(div ((V.length file)-3) 3)) :: IO Int)
                   let findex' = (findex*3)
                       pixel2 = L.map (file!)[(findex*3), ((findex*3)+1), ((findex*3)+2)]
                       pixel = L.cycle pixel2
                       dist = sum' $ L.zipWith (-) (toList img') pixel
                       indexL = L.elemIndices (L.minimum dist) dist
                   index <-do i <- randomRIO(0,((L.length indexL)-1)):: IO Int
                              return (indexL !! i)
                   let result = vUpdate img' (fromList pixel2) index
                       filed = (V.take (findex') file) V.++ (V.drop (findex'+3) file)
                   return $ result V.++ filed
                   where
                        sum' :: [Int] -> [Int]
                        sum' (x1:x2:x3:xs) = (abs x1)+(abs x2)+(abs x3) :sum' xs
                        sum' _ = []

mainloop:: ViewPort -> Float -> Vector Int -> IO (Vector Int)
mainloop _ _ img = do let leng = V.length img 
                      img' <- if leng == (7500)
                                  then do putStr "STOP" ; return (img)
                                  else do print $(leng - 7500); im <- mainloop' img ;return (im)
                      hFlush stdout
                      return img'
                      

main :: IO()
main = do 
    input <- intialize
    args <- getArgs
    file <- loadImage (L.head args)
    let somdata = input V.++ file
    simulateIO (InWindow "test" (800, 800) (0, 0)) white 500 somdata showImg mainloop
    where 
        intialize :: IO (Vector Int)
        intialize = let rseed = getStdRandom $ randomR (0,255) :: IO Int
                    in V.replicateM (50*50*3) rseed
        
        loadImage :: FilePath ->IO (Vector Int)
        loadImage x = do img <- readJpeg x
                         let img' = case img of Right s -> (\(ImageYCbCr8 t) -> t) s
                             img'' = V.map (fromIntegral) $ imageData $ (convertImage :: Image PixelYCbCr8->Image PixelRGB8)img'
                         return img''
        
        showImg :: Vector Int -> IO Picture
        showImg img' = do let img = V.take 7500 img'
                              imgdata = Scale 16 16 $ fromImageRGB8 $ mImg img
                          return imgdata
                           
        mImg :: Vector Int -> (Image PixelRGB8)
        mImg z = Image{imageWidth=50,imageHeight=50,imageData=(V.map mColor z)}
          
        mColor :: Int-> (PixelBaseComponent PixelRGB8)
        mColor x = (fromIntegral x::Word8)
