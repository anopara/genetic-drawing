import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import string
import random
from IPython.display import clear_output

class GeneticDrawing:
    def __init__(self, img_path, seed=0, brushesRange=[[0.1, 0.3], [0.3, 0.7]]):
        self.original_img = cv2.imread(img_path)
        self.img_grey = cv2.cvtColor(self.original_img,cv2.COLOR_BGR2GRAY)
        self.img_grads = self._imgGradient(self.img_grey)
        self.myDNA = None
        self.seed = seed
        self.brushesRange = brushesRange
        self.sampling_mask = None
        
        #start with an empty black img
        self.imgBuffer = [np.zeros((self.img_grey.shape[0], self.img_grey.shape[1]), np.uint8)]
        
    def generate(self, stages=10, generations=100, brushstrokesCount=10, show_progress_imgs=True):
        for s in range(stages):
            #initialize new DNA
            if self.sampling_mask is not None:
                sampling_mask = self.sampling_mask
            else:
                sampling_mask = self.create_sampling_mask(s, stages)
            self.myDNA = DNA(self.img_grey.shape, 
                             self.img_grads, 
                             self.calcBrushRange(s, stages), 
                             canvas=self.imgBuffer[-1], 
                             sampling_mask=sampling_mask)
            self.myDNA.initRandom(self.img_grey, brushstrokesCount, self.seed + time.time() + s)
            #evolve DNA
            for g in range(generations):
                self.myDNA.evolveDNASeq(self.img_grey, self.seed + time.time() + g)
                clear_output(wait=True)
                print("Stage ", s+1, ". Generation ", g+1, "/", generations)
                if show_progress_imgs is True:
                    #plt.imshow(sampling_mask, cmap='gray')
                    plt.imshow(self.myDNA.get_cached_image(), cmap='gray')
                    plt.show()
            self.imgBuffer.append(self.myDNA.get_cached_image())
        return self.myDNA.get_cached_image()
    
    def calcBrushRange(self, stage, total_stages):
        return [self._calcBrushSize(self.brushesRange[0], stage, total_stages), self._calcBrushSize(self.brushesRange[1], stage, total_stages)]
        
    def set_brush_range(self, ranges):
        self.brushesRange = ranges
        
    def set_sampling_mask(self, img_path):
        self.sampling_mask = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2GRAY)
        
    def create_sampling_mask(self, s, stages):
        percent = 0.2
        start_stage = int(stages*percent)
        sampling_mask = None
        if s >= start_stage:
            t = (1.0 - (s-start_stage)/max(stages-start_stage-1,1)) * 0.25 + 0.005
            sampling_mask = self.calc_sampling_mask(t)
        return sampling_mask
        
    '''
    we'd like to "guide" the brushtrokes along the image gradient direction, if such direction has large magnitude
    in places of low magnitude, we allow for more deviation from the direction. 
    this function precalculates angles and their magnitudes for later use inside DNA class
    '''
    def _imgGradient(self, img):
        #convert to 0 to 1 float representation
        img = np.float32(img) / 255.0 
        # Calculate gradient 
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #normalize magnitudes
        mag /= np.max(mag)
        #lower contrast
        mag = np.power(mag, 0.3)
        return mag, angle
    
    def calc_sampling_mask(self, blur_percent):
        img = np.copy(self.img_grey)
        # Calculate gradient 
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #calculate blur level
        w = img.shape[0] * blur_percent
        if w > 1:
            mag = cv2.GaussianBlur(mag,(0,0), w, cv2.BORDER_DEFAULT)
        #ensure range from 0-255 (mostly for visual debugging, since in sampling we will renormalize it anyway)
        scale = 255.0/mag.max()
        return mag*scale
        
    
    def _calcBrushSize(self, brange, stage, total_stages):
        bmin = brange[0]
        bmax = brange[1]
        t = stage/max(total_stages-1, 1)
        return (bmax-bmin)*(-t*t+1)+bmin


def util_sample_from_img(img):
    #possible positions to sample
    pos = np.indices(dimensions=img.shape)
    pos = pos.reshape(2, pos.shape[1]*pos.shape[2])
    img_flat = np.clip(img.flatten() / img.flatten().sum(), 0.0, 1.0)
    return pos[:, np.random.choice(np.arange(pos.shape[1]), 1, p=img_flat)]

class DNA:

    def __init__(self, bound, img_gradient, brushstrokes_range, canvas=None, sampling_mask=None):
        self.DNASeq = []
        self.bound = bound
        
        #CTRLS
        self.minSize = brushstrokes_range[0] #0.1 #0.3
        self.maxSize = brushstrokes_range[1] #0.3 # 0.7
        self.maxBrushNumber = 4
        self.brushSide = 300 #brush image resolution in pixels
        self.padding = int(self.brushSide*self.maxSize / 2 + 5)
        
        self.canvas = canvas
        
        #IMG GRADIENT
        self.imgMag = img_gradient[0]
        self.imgAngles = img_gradient[1]
        
        #OTHER
        self.brushes = self.preload_brushes('brushes/watercolor/', self.maxBrushNumber)
        self.sampling_mask = sampling_mask
        
        #CACHE
        self.cached_image = None
        self.cached_error = None
        
    def preload_brushes(self, path, maxBrushNumber):
        imgs = []
        for i in range(maxBrushNumber):
            imgs.append(cv2.imread(path + str(i) +'.jpg'))
        return imgs
    
    def gen_new_positions(self):
        if self.sampling_mask is not None:
            pos = util_sample_from_img(self.sampling_mask)
            posY = pos[0][0]
            posX = pos[1][0]
        else:
            posY = int(random.randrange(0, self.bound[0]))
            posX = int(random.randrange(0, self.bound[1]))
        return [posY, posX]
     
    def initRandom(self, target_image, count, seed):
        #initialize random DNA sequence
        for i in range(count):
            #random color
            color = random.randrange(0, 255)
            #random size
            random.seed(seed-i+4)
            size = random.random()*(self.maxSize-self.minSize) + self.minSize
            #random pos
            posY, posX = self.gen_new_positions()
            #random rotation
            '''
            start with the angle from image gradient
            based on magnitude of that angle direction, adjust the random angle offset.
            So in places of high magnitude, we are more likely to follow the angle with our brushstroke.
            In places of low magnitude, we can have a more random brushstroke direction.
            '''
            random.seed(seed*i/4.0-5)
            localMag = self.imgMag[posY][posX]
            localAngle = self.imgAngles[posY][posX] + 90 #perpendicular to the dir
            rotation = random.randrange(-180, 180)*(1-localMag) + localAngle
            #random brush number
            brushNumber = random.randrange(1, self.maxBrushNumber)
            #append data
            self.DNASeq.append([color, posY, posX, size, rotation, brushNumber])
        #calculate cache error and image
        self.cached_error, self_cached_image = self.calcTotalError(target_image)
        
    def get_cached_image(self):
        return self.cached_image
            
    def calcTotalError(self, inImg):
        return self.__calcError(self.DNASeq, inImg)
        
    def __calcError(self, DNASeq, inImg):
        #draw the DNA
        myImg = self.drawAll(DNASeq)

        #compare the DNA to img and calc fitness only in the ROI
        diff1 = cv2.subtract(inImg, myImg) #values are too low
        diff2 = cv2.subtract(myImg,inImg) #values are too high
        totalDiff = cv2.add(diff1, diff2)
        totalDiff = np.sum(totalDiff)
        return (totalDiff, myImg)
            
    def draw(self):
        myImg = self.drawAll(self.DNASeq)
        return myImg
        
    def drawAll(self, DNASeq):
        #set image to pre generated
        if self.canvas is None: #if we do not have an image specified
            inImg = np.zeros((self.bound[0], self.bound[1]), np.uint8)
        else:
            inImg = np.copy(self.canvas)
        #apply padding
        p = self.padding
        inImg = cv2.copyMakeBorder(inImg, p,p,p,p,cv2.BORDER_CONSTANT,value=[0,0,0])
        #draw every DNA
        for i in range(len(DNASeq)):
            inImg = self.__drawDNA(DNASeq[i], inImg)
        #remove padding
        y = inImg.shape[0]
        x = inImg.shape[1]
        return inImg[p:(y-p), p:(x-p)]       
        
    def __drawDNA(self, DNA, inImg):
        #get DNA data
        color = DNA[0]
        posX = int(DNA[2]) + self.padding #add padding since indices have shifted
        posY = int(DNA[1]) + self.padding
        size = DNA[3]
        rotation = DNA[4]
        brushNumber = int(DNA[5])

        #load brush alpha
        brushImg = self.brushes[brushNumber]
        #resize the brush
        brushImg = cv2.resize(brushImg,None,fx=size, fy=size, interpolation = cv2.INTER_CUBIC)
        #rotate
        brushImg = self.__rotateImg(brushImg, rotation)
        #brush img data
        brushImg = cv2.cvtColor(brushImg,cv2.COLOR_BGR2GRAY)
        rows, cols = brushImg.shape
        
        #create a colored canvas
        myClr = np.copy(brushImg)
        myClr[:, :] = color

        #find ROI
        inImg_rows, inImg_cols = inImg.shape
        y_min = int(posY - rows/2)
        y_max = int(posY + (rows - rows/2))
        x_min = int(posX - cols/2)
        x_max = int(posX + (cols - cols/2))
        
        # Convert uint8 to float
        foreground = myClr[0:rows, 0:cols].astype(float)
        background = inImg[y_min:y_max,x_min:x_max].astype(float) #get ROI
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = brushImg.astype(float)/255.0
        

        try:
            # Multiply the foreground with the alpha matte
            foreground = cv2.multiply(alpha, foreground)
            
            # Multiply the background with ( 1 - alpha )
            background = cv2.multiply(np.clip((1.0 - alpha), 0.0, 1.0), background)
            # Add the masked foreground and background.
            outImage = (np.clip(cv2.add(foreground, background), 0.0, 255.0)).astype(np.uint8)
            
            inImg[y_min:y_max, x_min:x_max] = outImage
        except:
            print('------ \n', 'in image ',inImg.shape)
            print('pivot: ', posY, posX)
            print('brush size: ', self.brushSide)
            print('brush shape: ', brushImg.shape)
            print(" Y range: ", rangeY, 'X range: ', rangeX)
            print('bg coord: ', posY, posY+rangeY, posX, posX+rangeX)
            print('fg: ', foreground.shape)
            print('bg: ', background.shape)
            print('alpha: ', alpha.shape)
        
        return inImg

        
    def __rotateImg(self, img, angle):
        rows,cols, channels = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        return dst
        
              
    def __evolveDNA(self, index, inImg, seed):
        #create a copy of the list and get its child  
        DNASeqCopy = np.copy(self.DNASeq)           
        child = DNASeqCopy[index]
        
        #mutate the child
        #select which items to mutate
        random.seed(seed + index)
        indexOptions = [0,1,2,3,4,5]
        changeIndices = []
        changeCount = random.randrange(1, len(indexOptions)+1)
        for i in range(changeCount):
            random.seed(seed + index + i + changeCount)
            indexToTake = random.randrange(0, len(indexOptions))
            #move it the change list
            changeIndices.append(indexOptions.pop(indexToTake))
        #mutate selected items
        np.sort(changeIndices)
        changeIndices[:] = changeIndices[::-1]
        for changeIndex in changeIndices:
            if changeIndex == 0:# if color
                child[0] = int(random.randrange(0, 255))
                #print('new color: ', child[0])
            elif changeIndex == 1 or changeIndex == 2:#if pos Y or X
                child[1], child[2] = self.gen_new_positions()
                #print('new posY: ', child[1], ' / ', self.bound[0])
                #print('new posX: ', child[2],  ' / ', self.bound[1])  
            elif changeIndex == 3: #if size
                child[3] = random.random()*(self.maxSize-self.minSize) + self.minSize
                #print('new size: ', child[3])
            elif changeIndex == 4: #if rotation
                #print("trying to mutate rotatino with child[1]", child[1], " and child[2] ", child[2])
                localMag = self.imgMag[int(child[1])][int(child[2])]
                localAngle = self.imgAngles[int(child[1])][int(child[2])] + 90 #perpendicular
                child[4] = random.randrange(-180, 180)*(1-localMag) + localAngle
                #print('new rot: ', child[4])
            elif changeIndex == 5: #if  brush number
                child[5] = random.randrange(1, self.maxBrushNumber)
                #print('new brush: ', child[5])
        #if child performs better replace parent
        #print('---\n', 'newchild: \n', child)
        child_error, child_img = self.__calcError(DNASeqCopy, inImg)
        if  child_error < self.cached_error:
            #print('mutation!', changeIndices)
            self.DNASeq[index] = child[:]
            self.cached_image = child_img
            self.cached_error = child_error
        
    def evolveDNASeq(self, inImg, seed):
        for i in range(len(self.DNASeq)):
            self.__evolveDNA(i, inImg, seed)
