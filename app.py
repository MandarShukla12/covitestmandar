import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER']= os.path.join('static', 'photos')
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'png']
data_transforms =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def predict(filename):
    img = Image.open('./static/photos/' + filename).convert('RGB')
    img_transformed = data_transforms(img)
    model = torch.load('.\covidmodel')
    outputs = model(img_transformed.unsqueeze(0))
    _, preds = torch.max(outputs, 1)
    return preds.detach().numpy() == 1


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
def view_home():
    covidno_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'NoCovid (41).jpg')
    covidyes_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Covid (41).jpg')
    return render_template("home.html", title="Home", covid_image=covidyes_filename, normal_image=covidno_filename)

@app.route("/cnn")
def view_cnn():
    cnn_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'CNN_Architecture.jpeg')
    return render_template("cnn.html", title="CNN", cnn_image= cnn_filename )

@app.route("/model")
def view_model():
    notebook_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Covid19_Predictions.html')
    return render_template("model.html", title="Model Performance", Notebook= notebook_filename)

@app.route('/prediction', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      file = request.files['file']
      if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename=filename))
   return render_template("prediction_upload.html", title="Prediction")


@app.route('/show/<filename>')
def uploaded_file(filename):
    pred=predict(filename)
    return render_template('prediction.html', filename=filename, result=pred)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)




