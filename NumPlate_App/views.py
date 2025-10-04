from django.shortcuts import render,redirect
from django.views import View
from django.contrib import messages
from django.contrib.sessions.models import Session
from .models import*
import cv2
import imutils
import numpy as np
import pytesseract
import tensorflow as tf


# Create your views here.

def Home(request):
	return render(request,"Home.html",{})

def Detect_Image(request):
	if request.method == "POST":
		img = request.FILES['img']
		print(str(img))
		obj = Image(img=img)
		obj.save()
		pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
		path = 'C:/Python Projects/Number_Plate_Detection/media/'+str(img)
		img = cv2.imread(path,cv2.IMREAD_COLOR)
		print(path)
		img = cv2.resize(img, (600,400) )
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
		gray = cv2.bilateralFilter(gray, 13, 15, 15) 

		edged = cv2.Canny(gray, 30, 200) 
		contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
		screenCnt = None

		for c in contours:
		    
		    peri = cv2.arcLength(c, True)
		    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
		 
		    if len(approx) == 4:
		        screenCnt = approx
		        break

		if screenCnt is None:
		    detected = 0
		    print ("No contour detected")
		else:
		     detected = 1

		if detected == 1:
		    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

		mask = np.zeros(gray.shape,np.uint8)
		new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
		new_image = cv2.bitwise_and(img,img,mask=mask)

		(x, y) = np.where(mask == 255)
		(topx, topy) = (np.min(x), np.min(y))
		(bottomx, bottomy) = (np.max(x), np.max(y))
		Cropped = gray[topx:bottomx+1, topy:bottomy+1]

		text = pytesseract.image_to_string(Cropped, config='--psm 11')
		print("programming_fever's License Plate Recognition\n")
		print("Detected license plate Number is:",text)
		img = cv2.resize(img,(500,300))
		Cropped = cv2.resize(Cropped,(400,200))
		cv2.imshow('car',img)
		cv2.imshow('Cropped',Cropped)

		cv2.waitKey(0)
		cv2.destroyAllWindows()
		return redirect('/')
	else:
		return render(request,"Detect_Image.html",{})

def Detect_Video(request):
	if request.method == "POST":
		video = request.FILES['video']
		print(video)
		obj = Image(img=video)
		obj.save()
		frameWidth = 640    #Frame Width
		franeHeight = 480   # Frame Height
		pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
		plateCascade = cv2.CascadeClassifier("C:/Python Projects/Number_Plate_Detection/haarcascade_russian_plate_number.xml")
		minArea = 500

		cap = cv2.VideoCapture("C:/Python Projects/Number_Plate_Detection/media/images/"+str(video))

		cap.set(3,frameWidth)
		cap.set(4,franeHeight)
		cap.set(10,150)
		count = 0

		while True:
		   	success , img  = cap.read()
		   	frame="frame"
		   	print(success)
		   	print(img)
		   	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		   	numberPlates = plateCascade .detectMultiScale(imgGray, 1.1, 4)
		   	for (x,y,w,h) in numberPlates:
		   		wT,hT,cT=img.shape
		   		a,b=(int(0.02*wT),int(0.02*hT))
		   		plate=img[y+a:y+h-a,x+b:x+w-b,:]
		   		#make the img more darker to identify LPR
		   		kernel=np.ones((1,1),np.uint8)
		   		plate=cv2.dilate(plate,kernel,iterations=1)
		   		plate=cv2.erode(plate,kernel,iterations=1)
		   		plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
		   		(thresh,plate)=cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)
		   		#read the text on the plate
		   		read=pytesseract.image_to_string(plate)
		   		read=''.join(e for e in read if e.isalnum())
		   		stat=read[0:2]
		   		cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
		   		cv2.rectangle(img,(x-1,y-40),(x+w+1,y),(51,51,255),-1)
		   		cv2.putText(img,read,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
		   		cv2.imshow("plate",plate)
		   	cv2.imwrite("Result.png",img)
		   	cv2.imshow("Result",img)
		   	if cv2.waitKey(1) & 0xFF ==ord('s'):
		   		cv2.imwrite("C:/Python Projects/Number_Plate_Detection/media/images"+str(count)+".jpg",imgRoi)
		   		cv2.imwrite("C:/Python Projects/Number_Plate_Detection/media/images"+str(frame)+".jpg",img)
		   		cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
		   		cv2.putText(img,"Scan Saved",(15,265),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
		   		cv2.imshow("Result",img)
		   	if cv2.waitKey(1) & 0xFF == ord('q'):
		   		break    	
		cap.release()
		cv2.destroyAllWindows()
		return render(request,"Detect_Video.html",{})
		        # count+=1
	else:
		return render(request,"Detect_Video.html",{})

def Real_Time(request):
	if request.method == "POST":
		frameWidth = 640    #Frame Width
		franeHeight = 480   # Frame Height
		pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
		plateCascade = cv2.CascadeClassifier("C:/Python Projects/Number_Plate_Detection/haarcascade_russian_plate_number.xml")
		minArea = 500

		cap =cv2.VideoCapture(0)
		cap.set(3,frameWidth)
		cap.set(4,franeHeight)
		cap.set(10,150)
		count = 0

		while True:
		    success , img  = cap.read()
		    frame="frame"
		    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		    numberPlates = plateCascade .detectMultiScale(imgGray, 1.1, 4)

		    for (x,y,w,h) in numberPlates:
		    	wT,hT,cT=img.shape
		    	a,b=(int(0.02*wT),int(0.02*hT))
		    	plate=img[y+a:y+h-a,x+b:x+w-b,:]
		    	#make the img more darker to identify LPR
		    	kernel=np.ones((1,1),np.uint8)
		    	plate=cv2.dilate(plate,kernel,iterations=1)
		    	plate=cv2.erode(plate,kernel,iterations=1)
		    	plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
		    	(thresh,plate)=cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)
		    	#read the text on the plate
		    	read=pytesseract.image_to_string(plate)
		    	read=''.join(e for e in read if e.isalnum())
		    	stat=read[0:2]
		    	cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
		    	cv2.rectangle(img,(x-1,y-40),(x+w+1,y),(51,51,255),-1)
		    	cv2.putText(img,read,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
		    	cv2.imshow("plate",plate)
		    cv2.imwrite("Result.png",img)
		    cv2.imshow("Result",img)
		    if cv2.waitKey(1) & 0xFF ==ord('s'):
		        cv2.imwrite("C:/Python Projects/Number_Plate_Detection/images"+str(count)+".jpg",imgRoi)
		        cv2.imwrite("C:/Python Projects/Number_Plate_Detection/images"+str(frame)+".jpg",img)
		        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
		        cv2.putText(img,"Scan Saved",(15,265),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
		        cv2.imshow("Result",img)
		    if cv2.waitKey(1) & 0xFF == ord('q'):
	        	break    	
		cap.release()
		cv2.destroyAllWindows()
		return render(request,"Real_Time.html",{})
		        # count+=1
	else:
		return render(request,"Real_Time.html",{})
















































def preprocess_image_for_prediction(image_path):
    # Read the image file
    image = cv2.imread(image_path)

    # Resize the image
    image = imutils.resize(image, width=500)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal with bilateral filter
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Find edges using Canny
    edged = cv2.Canny(gray, 170, 200)

    # Find contours based on edges
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]  # sort contours based on area

    NumberPlateCnt = None
    # Loop over contours to find the best possible approximate contour of the number plate
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            break

    if NumberPlateCnt is not None:
        # Find rotation angle
        opp = NumberPlateCnt[1][0][1] - NumberPlateCnt[0][0][1]
        hyp = np.linalg.norm(NumberPlateCnt[1][0] - NumberPlateCnt[0][0])
        sin_theta = opp / hyp
        theta = np.arcsin(sin_theta) * 57.2958

        # Rotate the image
        image_center = tuple(np.array(ROI.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, theta, 1.0)
        ROI = cv2.warpAffine(ROI, rot_mat, ROI.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Segment characters
        def find_contours(dimensions, img):
        	cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        	lower_width = dimensions[0]
        	upper_width = dimensions[1]
        	lower_height = dimensions[2]
        	upper_height = dimensions[3]

        	# Check largest 5 or  15 contours for license plate or character respectively
        	cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

        	ii = cv2.imread('contour.jpg')
        	x_cntr_list = []
        	target_contours = []
        	img_res = []
        	for cntr in cntrs :
        		# detects contour in binary image and returns the coordinates of rectangle enclosing it
		        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
		        
		        # checking the dimensions of the contour to filter out the characters by contour's size
		        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
		        	x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours
		        	char_copy = np.zeros((44,24))
		        	# extracting each character using the enclosing rectangle's coordinates.
		        	char = img[intY:intY+intHeight, intX:intX+intWidth]
		        	char = cv2.resize(char, (20, 40))
		        	cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
		        	plt.imshow(ii, cmap='gray')
		        	plt.title('Predict Segments')
		        	# Make result formatted for classification: invert colors
		        	char = cv2.subtract(255, char)
		        	# Resize the image to 24x44 with black border
		        	char_copy[2:42, 2:22] = char
		        	char_copy[0:2, :] = 0
		        	char_copy[:, 0:2] = 0
		        	char_copy[42:44, :] = 0
		        	char_copy[:, 22:24] = 0
		        	img_res.append(char_copy) # List that stores the character's binary image (unsorted)
		    plt.show()
		    # arbitrary function that stores sorted list of character indeces
		    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
		    img_res_copy = []
		    for idx in indices:
		        img_res_copy.append(img_res[idx])# stores character images according to their index
		    img_res = np.array(img_res_copy)

		    return img_res

        def segment_characters(image):
            # ... (complete the segment_characters function here)

        # Preprocess cropped license plate image for character segmentation
        img_lp = cv2.resize(ROI, (333, 75))
        img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]

        # Make borders white
        img_binary_lp[0:3, :] = 255
        img_binary_lp[:, 0:3] = 255
        img_binary_lp[72:75, :] = 255
        img_binary_lp[:, 330:333] = 255

        # Estimations of character contours sizes of cropped license plates
        dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]

        # Get segmented characters
        char_list = find_contours(dimensions, img_binary_lp)

        return ROI, char_list  # Return the preprocessed image and segmented characters

def predict_license_plate_number(image_path):
    # Load the pre-trained model
    loaded_model = tf.keras.models.load_model('C:/Users/Administrator/checkpoints/my_model')

    # Preprocess the image
    preprocessed_image = preprocess_image_for_prediction(image_path)
    img = preprocessed_image.reshape(1, 28, 28, 3)

    # Character mapping dictionary
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for i in range(len(preprocessed_image)):  # iterating over the characters
        img_ = cv2.resize(preprocessed_image[i], (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)  # preparing image for the model
        y_probs = loaded_model.predict(img)[0]  # predicting class probabilities
        y_ = np.argmax(y_probs)  # finding the class with the highest probability
        character = dic[y_]
        output.append(character)  # storing the result in a list

    plate_number = ''.join(output)

    return plate_number
