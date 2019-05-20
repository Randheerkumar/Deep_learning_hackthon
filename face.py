'''
Deep learning hackthon
17-05-2019
@Randheer kumar
'''


from mtcnn.mtcnn import MTCNN 
import cv2
import numpy as np
import glob
import os 



def my_mtcnn(img,folder,output):
	dir_name=output+"/"+folder
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	#os.makedirs(output+"/"+folder)
	cv2.imwrite(dir_name+"/"+"main"+".jpg",img)
	#img=cv2.imread('big.jpeg')  # reading images
	detector=MTCNN()
	#print(img.shape)
	ans=(detector.detect_faces(img))   #detecting faces
	no_of_faces=len(ans)

	arr=np.zeros((no_of_faces,4),int)
	for i in range(no_of_faces):
		temp_dict=ans[i]
		arr[i]=np.array(temp_dict['box'])

	#print(arr)
	y_max=img.shape[0]
	x_max=img.shape[1]

	#print("x and y max is :",x_max,y_max)
	for i in range(no_of_faces):
		tlx=int(arr[i][0])
		tly=int(arr[i][1])
		brx=int(arr[i][0]+arr[i][2])
		bry=int(arr[i][1]+arr[i][3])
		inc_x=int((arr[i][2]*10)/100)    		#   width:arr[][2]
		inc_y=int((arr[i][3]*10)/100)      		 #    height:arr[][3]

		#img_1 = 	cv2.rectangle(img,(tlx,tly),(brx,bry),(0,255,0),1)
		img_1=img[ max(0,tly-inc_y) : min(bry+inc_y,y_max) ,   max(0,tlx-inc_x):min(brx+inc_x,x_max),:]
		#img_1=plot(img,tlx,tly,brx,bry)
		#cv2.imshow('dark',img_1)
		cv2.imwrite(output+"/"+folder+"/"+str(tlx)+"_"+str(tly)+"_"+str(brx)+"_"\
			+str(bry)+"_"+str(i)+".png",img_1)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()



#dataset="face_dataset"    #folder containing images
#output="face_dataset_result"      #folder containing 

def crop(dataset,output):
	k=0
	for filename in os.listdir(dataset):
		img = cv2.imread(os.path.join(dataset,filename))
		new_dir="img_"+str(k)
		k+=1
		if img is not None:
			my_mtcnn(img,new_dir,output)
