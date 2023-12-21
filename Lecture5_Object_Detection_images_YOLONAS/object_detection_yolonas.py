import  cv2
import torch

from super_gradients.training import  models

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = models.get('yolo_nas_m', pretrained_weights="coco").to(device)

out = model.predict("../Images/image3.png")

out.show()