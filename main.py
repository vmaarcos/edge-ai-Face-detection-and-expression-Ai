from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

# Inicializa o aplicativo Flask
app = Flask(__name__)

# Carrega o modelo treinado uma vez quando o servidor inicia
model = tf.keras.models.load_model('emotion_model.h5')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializa a câmera
camera = cv2.VideoCapture(0)

def generate_frames():
    """Função geradora que captura frames da câmera, processa e os retorna como JPEGs."""
    while True:
        # Lê o frame da câmera
        success, frame = camera.read()
        if not success:
            break
        else:
            # Converte para grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detecta rostos
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Extrai o rosto
                roi_gray = gray[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi_gray, (48, 48))
                
                # Prepara para o modelo
                roi = roi_resized.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)
                
                # Predição
                prediction = model.predict(roi)
                emotion_index = np.argmax(prediction)
                emotion_label = emotions[emotion_index]
                
                # Desenha o retângulo e o texto no frame original (colorido)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Codifica o frame em formato JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Retorna o frame no formato de stream multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Rota principal que renderiza a página HTML."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Rota que fornece o stream de vídeo."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Inicia o servidor Flask
    # O host='0.0.0.0' torna o servidor acessível na sua rede local
    app.run(host='0.0.0.0', port=5000, debug=True)