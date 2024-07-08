import cv2
from paddleocr import PaddleOCR, draw_ocr

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # use the appropriate language

# Initialize video capture (0 for default camera, or provide stream URL)
cap = cv2.VideoCapture(0)

# Path to a valid TTF font file
font_path = '/home/shusrith/Downloads/Roboto/Roboto-Regular.ttf'  # Change this to a valid font path

# Set to keep track of already detected texts
detected_texts = set()
frame_count = 0 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count == 30:
        frame_count = 0
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = ocr.ocr(rgb_frame, cls=True)

        if len(result) > 0:
            for line in result:
                try:
                    for item in line:
                        text = item[1][0]

                        if text not in detected_texts:
                            print(text)
                            detected_texts.add(text)
                            boxes = item[0]
                            score = item[1][1]
                            frame = draw_ocr(frame, [boxes], [text], [score], font_path=font_path)
                except:
                    pass

        cv2.imshow('Text Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
