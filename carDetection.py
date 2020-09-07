import cv2

video=cv2.VideoCapture('traffic.mp4')

carCaseCase=cv2.CascadeClassifier('cars.xml')

while True:
	ret,frame=video.read()
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cars=carCaseCase.detectMultiScale(gray, 1.1, 9)
	for x,y,w,h in cars:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255),2)
		cv2.rectangle(frame, (x,y-40),(x+w, y), (50,50,255),-2)
		cv2.putText(frame,"Car",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.75, (255,255,255),2,cv2.LINE_AA)
	frame=cv2.resize(frame, (600,400))
	cv2.imshow("Car Detection",frame)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break
video.release()
cv2.destroyAllWindows()
