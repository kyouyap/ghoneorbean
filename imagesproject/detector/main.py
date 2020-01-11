import base64
import io
import cv2
import keras
import numpy as np
from PIL import Image
from keras.backend import tensorflow_backend as backend
from django.conf import settings

def detect(upload_image):
    result_name = upload_image.name
    result_list = []
    result_img = ''
    # 設定からカスケードファイルのパスを取得
    cascade_file_path = settings.CASCADE_FILE_PATH
    # 設定からモデルファイルのパスを取得

    model_file_path = settings.MODEL_FILE_PATH
    # kerasでモデルを読み込む

    faceCascade = cv2.CascadeClassifier(cascade_file_path)
    model = keras.models.load_model(model_file_path)
    # アップロードされた画像ファイルをメモリ上でOpenCVのimageに格納
    image = np.asarray(Image.open(upload_image))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 画像をOpenCVのBGRからRGB変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 画像をRGBからグレースケール変換
    face = faceCascade.detectMultiScale(gray, 1.1, 3)
    if len(face) > 0:
        for rect in face:
            # 顔認識部分を赤線で囲み保存(今はこの部分は必要ない)
            # cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=1)
            # cv2.imwrite('detected.jpg', img)
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            # print(rect)
            # print(x,y,w,h)
            image_rgb=image_rgb[y:y+h,x:x+w]
            # face_detect_count = face_detect_count + 1
    else:
        #print('image' + str(i) + ':NoFace')
        is_success, img_buffer = cv2.imencode(".png", image_rgb)
        if is_success:
            # 画像をインメモリのバイナリストリームに流し込む
            io_buffer = io.BytesIO(img_buffer)
            # インメモリのバイナリストリームからBASE64エンコードに変換
            result_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'", "")
        resslt_list.append("error:顔を認識できませんでした")
        return (result_list, result_name, result_img)
    # カスケードファイルの読み込み
    print(cascade_file_path)
    cascade = cv2.CascadeClassifier(cascade_file_path)
    # OpenCVを利用して顔認識
    # face_list = cascade.detectMultiScale(face, scaleFactor=1.1,minNeighbors=5, minSize=(250, 250))

    print("予測2")
    # 顔が１つ以上検出できた場合
    # print(len(face_list))
    # if len(face_list) > 0:

    # 認識した顔の切り抜き
    # face_image = image_rgb[ypos:ypos+height, xpos:xpos+width]
    # # 切り抜いた顔が小さすぎたらスキップ
    # if face_image.shape[0] < 250 or face_image.shape[1] < 250:
    #     continue
    # 認識した顔のサイズ縮小

    face_image=image_rgb
    face_image = cv2.resize(face_image, (250, 250))
    # 認識した顔のまわりを赤枠で囲む
    # cv2.rectangle(image_rgb, (xpos, ypos),
    #               (xpos+width, ypos+height), (0, 0, 250), thickness=2)
    # 認識した顔を1枚の画像を含む配列に変換
    face_image = np.expand_dims(face_image, axis=0)
    # 認識した顔から名前を特定
    predicted = model.predict(face_image)
    print(predicted)
    # 結果
    name = ""

    result = f"カルロスゴーンの可能性:{predicted[0][0]*100:.3f}% / ミスタービーンの可能性:{predicted[0][1]*100:.3f}%"
    name_number_label = np.argmax(predicted)
    if name_number_label == 0:
        name = "Gohne"
    elif name_number_label == 1:
        name = "Bean"

    print("予測3")
    # 認識した顔に名前を描画
    count=1
    # cv2.putText(image_rgb, f"{count}. {name}", (xpos, ypos+height+20),
    #             cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 250), 2)
    # 結果をリストに格納
    result_list.append(result)
    count = count + 1

    # 画像をPNGに変換
    is_success, img_buffer = cv2.imencode(".png", image_rgb)
    if is_success:
        # 画像をインメモリのバイナリストリームに流し込む
        io_buffer = io.BytesIO(img_buffer)
        # インメモリのバイナリストリームからBASE64エンコードに変換
        result_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'", "")

    # tensorflowのバックエンドセッションをクリア
    backend.clear_session()
    # 結果を返却
    return (result_list, result_name, result_img)

# def detect_who(model, face_image):
    # 予測

    # predicted = model.predict(face_image)
    # print(predicted)
    # # 結果
    # name = ""
    # result = f"本田 翼 の可能性:{predicted[0][0]*100:.3f}% / 佐倉 綾音 の可能性:{predicted[0][1]*100:.3f}%"
    # name_number_label = np.argmax(predicted)
    # if name_number_label == 0:
    #     name = "Honda Tsubasa"
    # elif name_number_label == 1:
    #     name = "Sakura Ayane"
    # return (name, result)
