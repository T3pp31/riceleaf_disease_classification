

import os
import torch
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
from model import build_model
from config import CLASS_NAMES  # クラス名リストをインポート

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# アップロードされたファイルを返すエンドポイントを追加
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    JSDoc: アップロードされたファイルを返します。
    引数: filename - 表示したいファイル名（str）
    戻り値: ファイルのレスポンス
    注意: セキュリティのため、ファイル名は必ず検証・サニタイズしてください。
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Load the model
model_path = 'models/woven-cosmos-4-epoch15-acc99.83.pth'
model = build_model(pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 画像のクラスを予測し、クラス名に変換して返す
            image = Image.open(filepath).convert('RGB')
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                predicted_index = predicted.item()  # 予測インデックス
                # インデックスからクラス名へ変換
                prediction = CLASS_NAMES[predicted_index] if 0 <= predicted_index < len(CLASS_NAMES) else 'Unknown'

            return render_template('result.html', filename=filename, prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    # JSDoc: アプリケーションをローカルネットワーク上の他の端末からもアクセスできるように起動します。
    # 引数: host='0.0.0.0' で全てのネットワークインターフェースをバインドします。
    # 戻り値: なし
    # 注意: セキュリティのため、本番環境では適切なWAFや認証を導入してください。
    app.run(host='0.0.0.0', debug=True)
