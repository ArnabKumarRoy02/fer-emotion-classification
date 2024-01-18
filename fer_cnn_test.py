import torch
from fer_cnn import EmotionClassifierCNN
from torchvision import transforms
from PIL import Image

# Load the trained model
model = EmotionClassifierCNN()
model.load_state_dict(torch.load('emotion-classifier.pth'))
model.eval()

# Image Preprocessing transformation
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Load the image and preprocess
img = Image.open('fer-2013/test/happy/PublicTest_98860460.jpg')
preprocessed_img = preprocess(img)

# Make predictions
with torch.no_grad():
    output = model(preprocessed_img.unsqueeze(0))

# Get the predicted class
_, pred = torch.max(output, 1)

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
predicted_class_name = class_names[pred.item()]

# Print the predicted class name
print(f'Predicted class is: {predicted_class_name}')