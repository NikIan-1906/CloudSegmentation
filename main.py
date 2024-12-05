import cv2
import UNET
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_transform = A.Compose([A.Resize(300, 300),
                           A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255),
                           ToTensorV2()])
trained_model = UNET.UNET(in_chnls = 3, n_classes = 1)
trained_model.load_state_dict(torch.load("unet_scratch.pth", map_location="cpu"))
trained_model = trained_model.to(device)

def find_clouds(image):
    img = image
    test_image = test_transform(image=img)
    img = test_image["image"].unsqueeze(0)
    img = img.to(device)
    pred = trained_model(img)

    mask = pred.squeeze(0).cpu().detach().numpy()
    mask = mask.transpose(1, 2, 0)
    mask[mask <= 1] = 0
    mask[mask > 1] = 1

    return mask

while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(33) & 0xFF == 32:
        img = frame
        #cv2.imshow("img", img)
        mask = find_clouds(img)
        cv2.imshow("clouds", mask)

    if cv2.waitKey(33) & 0xFF == ord("q") or not ret:
        break

    cv2.imshow("frame", frame)

cap.release()
cv2.destroyAllWindows()
