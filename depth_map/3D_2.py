import cv2
import time
import numpy as np
from multiprocessing import Pool
from openpyxl import Workbook # Used for writing data into an Excel file
# from sklearn.preprocessing import normalize


def doWork(st): #j=1 es izquierdo , j=2 es derecho
    grayL = st[0] 
    grayR = st[1]
    j = st[2]

    # Create StereoSGBM and prepare all parameters
    window_size = 5
    min_disp = 2
    num_disp = 130-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 5,
        preFilterCap = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2)

    # Used for the filtered image
    if j == 1 :
        disp= stereo.compute(grayL,grayR)
    
    if j == 2 :
        stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time
        disp= stereoR.compute(grayR,grayL)

    return disp



print("Reading parameters ......")
cv_file = cv2.FileStorage("params_py.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 5
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    preFilterCap = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000#80000
sigma = 1.8 #1.8
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

cv2.useOptimized()
wb=Workbook()
ws=wb.active  

# write into the excell worksheet

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        """
				p p p
				p p p
				p p p
        """
        average=0
        for u in range (-1,2):     # (-1 0 1)
            for v in range (-1,2): # (-1 0 1)
                average += disp[y+u,x+v]
        average=average/9
        #Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        #Distance= np.around(Distance*0.01,decimals=2)
        #print('Distance: '+ str(Distance)+' m')
        print('Average: '+ str(average))
        counterdist = int(input("ingresa distancia (cm): "))
        ws.append([counterdist, average])




# # Computo para el stereo
if __name__ == "__main__":
    with Pool(processes=2) as pool :
        startTime = time.time()
       # imgL = cv2.imread("chessboard-L%d.png"%2)
       # imgR = cv2.imread("chessboard-R%d.png"%2)

        imgL = cv2.imread("chessboard-L7.png")
        imgR = cv2.imread("chessboard-R7.png")
        Left_nice= cv2.remap(imgL,Left_Stereo_Map_x,Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        Right_nice= cv2.remap(imgR,Right_Stereo_Map_x,Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        grayR= Right_nice
        grayL= Left_nice

        # Compute the 2 images for the Depth_image
        # Run the pool in multiprocessing
        st1 = (grayL,grayR,1 )
        st2 = (grayL,grayR,2 )

        # Computo para el stereo
        disp , dispR = pool.map(doWork, (st1,st2))
            
        dispL= disp

        dispL= np.int16(dispL)
        dispR= np.int16(dispR)
        
        # Using the WLS filter
        filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)

        # Change the Color of the Picture into an Ocean Color_Map
        filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_DEEPGREEN    ) 

        cv2.imshow('Filtered Color Depth',filt_Color)

        # Draw Red lines
        for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
            Left_nice[line*20,:]= (0,0,255)
            Right_nice[line*20,:]= (0,0,255)
      
        cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
        
        # Mouse click
        cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)
        
        #mark the end time
        endTime = time.time()    
        
        pool.terminate()
        pool.join() 
        #calculate the total time it took to complete the work
        workTime =  endTime - startTime
        
        #print results
    print ("The job took " + str(workTime) + " sconds to complete")

    cv2.waitKey(0)







