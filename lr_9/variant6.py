import cv2
import numpy as np
from google.colab.patches import cv2_imshow


# Load an image
def load_image(path):
    return cv2.imread(path)


# Distort image by adding a sickle shape and random noise
def distort_image(image):
    # Add a sickle shape
    cv2.ellipse(image, (image.shape[1]//2, image.shape[0]//2), (100, 50), 0, 0, 180, (0, 0, 255), -1)
   
    # Add random noise
    noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
    return cv2.add(image, noise)


# Enhance image quality
def enhance_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    return cv2.equalizeHist(img_blur)


# Detect edges
def detect_edges(image):
    return cv2.Canny(image, 100, 200)


# Load and process an image
image = load_image('variant6.jpg')
distorted_image = distort_image(image)
enhanced_image = enhance_image(distorted_image)
edges = detect_edges(enhanced_image)
cv2_imshow(image)
cv2_imshow(distorted_image)
cv2_imshow(enhanced_image)
cv2_imshow(edges)
image_part2 = load_image('/content/variant6part2.jpg')
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()
from PIL import Image
from torchvision import transforms


im_pil = Image.fromarray(image_part2)
input_image = im_pil.convert('RGB')




preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')


with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
# Download ImageNet labels
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
