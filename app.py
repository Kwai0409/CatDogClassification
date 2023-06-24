from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# Load your pre-trained model
model = torch.load("model.pt")
model.eval()

# Define the transformation to be applied to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the classes for the prediction
classes = ['cat', 'dog']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file from the form
        file = request.files['file']
        # Read the image and apply the transformation
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.cuda()
        # Make the prediction
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            prediction = classes[predicted.item()]
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
