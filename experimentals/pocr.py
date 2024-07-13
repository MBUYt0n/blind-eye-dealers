import cv2
from paddleocr import PaddleOCR, draw_ocr

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)  # use the appropriate language

# Initialize video capture (0 for default camera, or provide stream URL)
cap = cv2.VideoCapture("/home/shusrith/vid.mp4")

# Path to a valid TTF font file
font_path = "/usr/share/fonts/truetype/fonts-gujr-extra/aakar-medium.ttf"  # Change this to a valid font path

# Set to keep track of already detected texts
# detected_texts = set()
frame_count = 0 
f = open("output.txt", "w")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count == 10:
        frame_count = 0
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = ocr.ocr(rgb_frame, cls=True)

        if len(result) > 0:
            for line in result:
                try:
                    a = ""
                    for item in line:
                        a += item[1][0]
                        # a += text + " "
                        # if text not in detected_texts:
                        #     # detected_texts.add(text)
                        # boxes = item[0]
                        # score = item[1][1]
                        # frame = draw_ocr(frame, [boxes], [a], [score], font_path=font_path)
                    f.write(a + "\n")
                except:
                    pass

        # cv2.imshow('Text Detection', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
# cv2.destroyAllWindows()
