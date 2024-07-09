import torch
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
).to(device)

cap = cv2.VideoCapture(0)

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)

        generated_ids = model.generate(pixel_values)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if text:
            print(text)
    # Ensure cv2.imshow() has a window name and correct argument for display
    cv2.imshow("Frame", frame)
    # Add a break condition on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture after exiting the loop
cap.release()
cv2.destroyAllWindows()
