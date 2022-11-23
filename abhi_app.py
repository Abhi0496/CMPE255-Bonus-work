import io
import json
from PIL import Image
from torchvision import models
from openvino.runtime import Core
from flask import Flask, jsonify, request
import torchvision.transforms as transforms


app = Flask(__name__)

## Loading model and labels
imagenet_class_index = json.load(open('_static/imagenet_class_index.json'))

tmodel = models.resnet34(pretrained=True)
tmodel.eval()

core = Core()
ovmodel = core.read_model(model='openvino_model/resnet34.xml')
compiled_model_ir = core.compile_model(model=ovmodel)
output_layer_ir = compiled_model_ir.output(0)

def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return my_transforms(image).unsqueeze(0)


def get_openvino_prediction(image):
    tensor = transform_image(image=image)
    img_np = tensor.numpy()
    print(img_np.shape)
    output = compiled_model_ir([img_np])[output_layer_ir][0]
    y_hat = output.argmax()
    predicted_idx = str(y_hat)
    return imagenet_class_index[predicted_idx]

def get_torch_prediction(image):
    tensor = transform_image(image=image)
    outputs = tmodel.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/torch', methods=['POST'])
def predict_torch():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        class_id, class_name = get_torch_prediction(image=image)
        return jsonify({'class_id': class_id, 'class_name': class_name})

@app.route('/openvino', methods=['POST'])
def predict_openvino():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        class_id, class_name = get_openvino_prediction(image=image)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()