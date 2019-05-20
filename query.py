#*********************************Hackthon*******************************************************
'''			                                                             						*									                   	   *
Hackthon:DeeP Learning                                                                          *
Task:Facial qury based system                                                                   *
                                                                                                *
'''#                                                                                            *
#************************************************************************************************

#***************************import the required libraries and files*****************************
import face
import os 
import facenet as fc
import tensorflow as tf
import cv2
from gender_detection.guess import main 
from guess_new import main as main_age
import shutil


#*********************this the FLAGS which will be used by gender and age detction network************
def flag(model_dir):
	#checkpoint
	tf.app.flags.DEFINE_string('model_dir',model_dir,
							   'Model directory (where training data lives)')

	tf.app.flags.DEFINE_string('class_type', 'gender',
							   'Classification type (age|gender)')


	tf.app.flags.DEFINE_string('device_id', '/cpu:0',
							   'What processing unit to execute inference on')

	tf.app.flags.DEFINE_string('filename','',
							   'File (Image) or File list (Text/No header TSV) to process')

	tf.app.flags.DEFINE_string('target', '',
							   'CSV file containing the filename processed along with best guess and score')

	tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
							  'Checkpoint basename')

	tf.app.flags.DEFINE_string('model_type', 'inception',
							   'Type of convnet')

	tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

	tf.app.flags.DEFINE_boolean('single_look', True, 'single look at the image or multiple crops')

	tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

	tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')

	FLAGS = tf.app.flags.FLAGS
	return FLAGS

#***************************loacate the persion in bounding box*********************************
def locate_person_bounding_bax(real_image,box,output_image_name,output_dir):
	img=cv2.imread(real_image)
	cordinate=box.split("_")
	x0,y0,x1,y1=int(cordinate[0]),int(cordinate[1]),int(cordinate[2]),int(cordinate[3])
	#print("box=",x0,y0,x1,y1)
	img=cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),1)
	output_image_name=output_image_name.split("/")
	output_img=output_dir+"/"+output_image_name[0]+".png"
	#print("path is:",output_img)
	#print("image=",img)
	cv2.imwrite(output_img,img)

#**********************this function mark the face with label 'M' or 'F'*************************
def detect_gender(img,box,gender):
	cordinate=box.split("_")
	x0,y0,x1,y1=int(cordinate[0]),int(cordinate[1]),int(cordinate[2]),int(cordinate[3])
	#print("box=",x0,y0,x1,y1)
	#text="Male"
	# if gender=='M':
	# 	print("in male of quotes	")

	if gender=='M':
		#print("in male")
		#fontScale=0.5, color=(0, 0, 255), thickness=2
		img=cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),1)
		img=cv2.putText(img,"M", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA) 
	else:
		img=cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),1)
		img=cv2.putText(img,"F", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),lineType=cv2.LINE_AA) 

	return img

#**********this function locate the bounding box of the reference face in each images in the gallery
def locate_person_bounding_box(img,box,ref_image):
	cordinate=box.split("_")
	x0,y0,x1,y1=int(cordinate[0]),int(cordinate[1]),int(cordinate[2]),int(cordinate[3])
	#print("box=",x0,y0,x1,y1)
	text=ref_image.split(".")[0]
	img=cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),1)
	img=cv2.putText(img,"found", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA) 

	return img


#********************mark each person with their age****************************************
def detect_age(img,box,age):
	cordinate=box.split("_")
	x0,y0,x1,y1=int(cordinate[0]),int(cordinate[1]),int(cordinate[2]),int(cordinate[3])
	#print("box=",x0,y0,x1,y1)
	text=age+" ys"
	img=cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),1)
	img=cv2.putText(img,text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), lineType=cv2.LINE_AA) 

	return img


################################path to the required model###################################
output_dir="/home/randheer/Desktop/6thsem/dl/hackthon/bounding_box"

model_dir_gender="/home/randheer/Desktop/6thsem/dl/hackthon/gender_detection/21936/"
FLAGS=flag(model_dir_gender)

model_dir_age="/home/randheer/Desktop/6thsem/dl/hackthon/22801/"
image_path="/home/randheer/Desktop/6thsem/dl/hackthon/bounding_box/img_3.png"

age="age"

s_crop="temp_dir"   #temporary directory for image dataset
ref_crop="ref_crop"

s=input("path to the gallery data:")
print("preprocessing gallary data...........................\n")
face.crop(s,s_crop)
i_face=0;i_gender=0;i_age=0

if os.path.exists(output_dir):
	shutil.rmtree(output_dir)
os.mkdir(output_dir)

#############################here the query system starts###############################################
while(1):
	case=int(input("0 : exit\n1 : face recognition\n2 : gender prediction\n3 : age perdiction\n"))

	if case==0:
		exit(0)

	elif case==1:
		i_face+=1
		ref=input("enter reference image path\n")
		print("preprocessing........................\n")
		output_dir1=output_dir+"/face_recognition"+str(i_face)
		os.mkdir(output_dir1)
		if os.path.exists(ref_crop):
			shutil.rmtree(ref_crop)

		os.mkdir(ref_crop)
		face.crop(ref,ref_crop)

		for check_folder in os.listdir(s_crop):
			join_folder=os.path.join(s_crop,check_folder)
			actual_image=os.path.join(join_folder,"main.jpg")
			img=cv2.imread(actual_image)
			for check_img in os.listdir(join_folder):
				base_img=os.path.join(join_folder,check_img)
				if check_img!="main.jpg":

					for ref_dir in os.listdir(ref_crop):
						ref_dir1=os.path.join(ref_crop,ref_dir)

						for ref_image in os.listdir(ref_dir1):
							#print("ref_image=",ref_image)
							#actual_image=os.path.join(ref_folder,"main.jpg")
							if ref_image != "main.jpg":
								ref_img1=os.path.join(ref_dir1,ref_image)

								#img_2=os.path.join(join_folder,check_img)
								#print("img1 and two is",ref_image,img_2)
								score=fc.verifyFace(ref_img1,base_img)
								if score<0.38:
									img=locate_person_bounding_box(img,check_img,ref_image)

								#print(score)

			im=check_folder.split("/")
			output_img=output_dir1+"/"+im[0]+".png"
			#print("output_path=",output_img,"img=",img)
			cv2.imwrite(output_img,img)

	
	elif case==2:
		i_gender+=1
		#ref=input("enter ref dataset\n")
		output_dir1=output_dir+"/Gender"+str(i_gender)	
		os.mkdir(output_dir1)

		for check_folder in os.listdir(s_crop):
			join_folder=os.path.join(s_crop,check_folder)
			actual_image=os.path.join(join_folder,"main.jpg")
			img=cv2.imread(actual_image)

			for check_img in os.listdir(join_folder):
				if check_img != "main.jpg":
					img_2=os.path.join(join_folder,check_img)
					gender=str(main(img_2,FLAGS))
					s=gender[2]
					img=detect_gender(img,check_img,s)
					#s=gender[2]
					#print("gender=",s," type=",type(s))

			im=check_folder.split("/")
			output_img=output_dir1+"/"+im[0]+".png"
			#print("path is:",output_img)
			#print("image=",img)
			cv2.imwrite(output_img,img)

	elif case==3 :
		i_age+=1
		#ref=input("enter ref dataset\n")
		output_dir1=output_dir+"/age"+str(i_age)
		os.mkdir(output_dir1)

		for check_folder in os.listdir(s_crop):
			join_folder=os.path.join(s_crop,check_folder)
			actual_image=os.path.join(join_folder,"main.jpg")
			img=cv2.imread(actual_image)

			for check_img in os.listdir(join_folder):
				
				if check_img != "main.jpg":
					img_2=os.path.join(join_folder,check_img)
					# gender=str(main(img_2,FLAGS))
					# s=gender[2]
					ag=main_age(model_dir_age,age,img_2,FLAGS)
					ag1=ag.split("'")
					ag2=ag1[1]
					ag2=ag2.split("(")[1]
					ag2=ag2.split(")")[0]
					ag2=ag2.split(",")
					ag3=ag2[0]+"-"+ag2[1]

					img=detect_age(img,check_img,ag3)

					#print("age=",ag2)

					#img=detect_gender(img,check_img,s)
					#s=gender[2]
					#print("gender=",s," type=",type(s))

			im=check_folder.split("/")
			output_img=output_dir1+"/"+im[0]+".png"
			#print("path is:",output_img)
			#print("image=",img)
			cv2.imwrite(output_img,img)

#*****************************************END**************************************************


# if case==1:
# 		ref=input("ref image path\n")
# 		face.crop(ref,ref_crop)

# 		for folder in os.listdir(ref_crop):  
# 			ref_folder=os.path.join(ref_crop,folder)

# 			for ref_image in os.listdir(ref_folder):
# 				actual_image=os.path.join(ref_folder,"main.jpg")
# 				if ref_image != "main.jpg":
# 					ref_img=os.path.join(ref_folder,ref_image)

# 					for check_folder in os.listdir(s_crop):
# 						join_folder=os.path.join(s_crop,check_folder)
# 						for check_img in os.listdir(join_folder):
# 							if check_img != "main.jpg":
# 								img_2=os.path.join(join_folder,check_img)
# 								#print("img1 and two is",ref_image,img_2)
# 								score=fc.verifyFace(ref_img,img_2)
# 								if score<0.37:
# 									locate_person_bounding_bax(actual_image,ref_image,ref_folder,output_dir)

# 								print(score)

# model_dir_age="/home/randheer/Desktop/6thsem/dl/hackthon/gender_detection/22801/"
# FLAGS_age=flag(model_dir_age)
# print("return main1=",main(model_dir_age,age,image_path,FLAGS))
# print("return main2=",main(model_dir_age,age,image_path,FLAGS))
#tf.reset_default_graph()
#print(main(image_path,FLAGS_age))    
# if __name__ == '__main__':

# image_path="/home/randheer/Desktop/6thsem/dl/hackthon/veeru.jpeg",
#output_dir="/home/randheer/Desktop/6thsem/dl/hackthon/bounding_box"
# f=main_age(image_path,FLAGS_age)
# # print(f)
# s=str(f)
# print(s)


# f=main(image_path,FLAGS)
# print(f)
# s=str(f)
# print(s)
		
		

