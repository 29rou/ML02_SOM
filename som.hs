import System.IO
import System.Environment (getArgs)
import Data.List as L
import Data.Vector.Storable as V
import Data.Word(Word8)
import System.Random(getStdRandom,randomR,randomRIO)
import Codec.Picture
import Codec.Picture.Types
import Graphics.Gloss(Picture(..),Display(..),white)
import Graphics.Gloss.Juicy(fromImageRGB8)
import Graphics.Gloss.Interface.IO.Simulate(simulateIO,ViewPort)

intialize :: IO (Vector Int)
intialize = let rseed = getStdRandom $ randomR (0,255) :: IO Int
            in V.replicateM (100*100*3) rseed

vExtract:: Vector Int -> Vector Int -> Int -> Vector Int
vExtract img ipixel n = let ext = (img !)
                            n' = n*3
                            (ir,ig,ib) = (ipixel!0, ipixel!1, ipixel!2)
                            (r,g,b) =(ext n',ext (n'+1),ext (n'+2))
                            rgb = fromList [((ir*1)+(r*1)),((ig*1)+(g*1)),((ib*1)+(b*1))]
                            pixel = V.map (\x -> div x 2 ) rgb
                            pixel' = V.map (\x -> if x>255 then 255 else x) pixel
                        in update_ img (fromList [n',n'+1,n'+2]) pixel'

vExtract2:: Vector Int -> Vector Int -> Int -> Vector Int
vExtract2 img ipixel n = let ext = (img !)
                             n' = n*3
                             (ir,ig,ib) = (ipixel!0, ipixel!1, ipixel!2)
                             (r,g,b) =(ext n',ext (n'+1),ext (n'+2))
                             rgb = fromList [((ir*1)+(r*4)),((ig*1)+(g*4)),((ib*1)+(b*4))]
                             pixel = V.map (\x -> div x 5 ) rgb
                             pixel' = V.map (\x -> if x>255 then 255 else x) pixel
                         in update_ img (fromList [n',n'+1,n'+2]) pixel'

vExtract3:: Vector Int -> Vector Int -> Int -> Vector Int
vExtract3 img ipixel n = let ext = (img !)
                             n' = n*3
                             (ir,ig,ib) = (ipixel!0, ipixel!1, ipixel!2)
                             (r,g,b) =(ext n',ext (n'+1),ext (n'+2))
                             rgb = fromList [(ir+(r*9)),(ig+(g*9)),(ib+(b*9))]
                             pixel = V.map (\x -> div x 10) rgb
                             pixel' = V.map (\x -> if x>255 then 255 else x) pixel
                         in update_ img (fromList [n',n'+1,n'+2]) pixel'

isqrt' :: Int -> Int -> Int
isqrt' x n = let x' = (fromIntegral x) ::Float
                 n' = (1 ::Float) / ((fromIntegral n)::Float)
                 y = truncate (x' ** n')
             in ((fromIntegral y) :: Int)

isqrt :: Int -> Int
isqrt = floor . sqrt . fromIntegral

coordinate :: Vector Int -> Vector Int -> Int ->Vector Int
coordinate img pixel index = let vExtract' n = vExtract n pixel
                                 vExtract'' n = vExtract2 n pixel
                                 vExtract''' n = vExtract3 n pixel
                                 mod' = ((mod index 100)/=)
                                 --indexList i = fromList [i,i+1,i+2]
                                 if' cod ind image = if cod then vExtract' image ind else image
                                 if'' cod ind image = if cod then vExtract'' image ind else image
                                 if''' cod ind image = if cod then vExtract''' image ind else image
                                 up = if' (index>=100) (index-100) 
                                 up' = if''' (index>=200) (index-200) 
                                 down = if' (index<=9899) (index+100) 
                                 down' = if''' (index<=9799) (index+200)
                                 right = if' (mod' 99) (index+1)
                                 right' = if''' ((mod' 99)&&(mod' 98)) (index+2)
                                 left = if' (mod' 0) (index-1)
                                 left' = if''' ((mod' 0)&&(mod' 1)) (index-2)
                                 upright = if'' (index>=100 && mod' 99) (index-99)
                                 upleft = if'' (index>=100 && mod' 0) (index-101)
                                 downright = if'' (index<=9899 && mod' 99) (index+101)
                                 downleft = if'' (index<=9899 && mod' 0) (index+99)
                                 img' = up $ down $ right $ left $ upright $ upleft $ downright $ downleft img
                              in up' $ down' $ right' $ left' img'

vUpdate :: Vector Int -> Vector Int -> Int ->Vector Int
vUpdate img pixel index = let pixeld = V.map (*4) pixel
                              ipix=fromList [img!(index*3),img!((index*3)+1),img!((index*3)+2)]
                              pixel' = V.map (\x -> div x 5) $ V.zipWith (+) pixeld ipix
                              pixel1 = V.map (\x -> if x>255 then 255 else x) pixel'
                              --change = up V.++up' V.++down v.++down' V.++ right V.++ right' V.++left V.++left' V.++upright V.++upleft V.++downright V.++downleft
                              --target = 
                              img2 = coordinate img pixel index
                              img3 = update_ img2 (fromList [index*3,(index*3)+1,(index*3)+2]) pixel1
                          in img3

mainloop:: ViewPort -> Float -> Vector Int -> IO (Vector Int)
mainloop _ _ img = do --rpixel <- V.replicateM 3 (getStdRandom $ randomR (0,255) :: IO Int)
                      --file <- loadImage "./got.jpg"
                      if V.length img == (30000)
                        then do putStrLn "STOP" ; text <- getLine; return text
                        else do return "OK"
                      let (img', file) = V.splitAt (30000) img
                          --(pixel3, filed) = V.splitAt 3 file
                      print $ V.length file
                      findex <- (getStdRandom $ randomR (0,(div ((V.length file)-3) 3)) :: IO Int)
                      let findex' = (findex)*3
                          pixel2 = [file!findex', file!(findex'+1), file!(findex'+2)]
                          --pixel2 = [pixel3!0, pixel3!1, pixel3!2]
                          pixel = fromList $ L.concat $ L.replicate 10000 pixel2
                          dist = sum' $ toList $ V.map (abs) $ V.zipWith (-) img' pixel
                          indexL = L.elemIndices (L.minimum dist) dist
                      index <-do i <- randomRIO(0,((L.length indexL)-1)):: IO Int
                                 return (indexL !! i)
                      let result = vUpdate img' (fromList pixel2) index
                          filed = (V.take (findex') file) V.++ (V.drop (findex'+3) file)
                      return $ result V.++ filed
                      where
                        sum' :: [Int] -> [Int]
                        sum' (x1:x2:x3:xs) = x1+x2+x3 :sum' xs
                        sum' _ = []

loadImage :: FilePath ->IO (Vector Int)
loadImage x = do img <- readJpeg x
                 let img' = case img of Right s -> (\(ImageYCbCr8 t) -> t) s
                     img'' = V.map (fromIntegral) $ imageData $ (convertImage :: Image PixelYCbCr8->Image PixelRGB8)img'
                 return img''
main :: IO()
main = do 
    hSetBuffering stdout NoBuffering
    input <- intialize
    args <- getArgs
    file <- loadImage (L.head args)
    let somdata = input V.++ file
    simulateIO (InWindow "test" (800, 800) (0, 0)) white 100 somdata showImg mainloop
    where 
        showImg :: Vector Int -> IO Picture
        showImg img' = do let img = V.take 30000 img'
                              imgdata = fromImageRGB8 $ mImg img
                          return $ Scale 8 8 imgdata
                           
        mImg :: Vector Int -> (Image PixelRGB8)
        mImg z = Image{imageWidth=100,imageHeight=100,imageData=(V.map mColor z)}
          
        mColor :: Int-> (PixelBaseComponent PixelRGB8)
        mColor x = (fromIntegral x::Word8)
