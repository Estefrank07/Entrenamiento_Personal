from flask import Flask, render_template, request, redirect, url_for
import pyrebase
import requests
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
from math import acos, degrees
from datetime import datetime 

Config = {
  "apiKey": "AIzaSyCygDml2jSUSaClrfuC3ETeVUfNu4CW_ds",
  "authDomain": "sistemaentrenamiento-54555.firebaseapp.com",
  "databaseURL": "https://sistemaentrenamiento-54555-default-rtdb.firebaseio.com",
  "projectId": "sistemaentrenamiento-54555",
  "storageBucket": "sistemaentrenamiento-54555.appspot.com",
  "messagingSenderId": "476633987540",
  "appId": "1:476633987540:web:a34e2b5be0f62a11fc7e41",
  "measurementId": "G-H53L40JK9D"
}

firebase = pyrebase.initialize_app(Config)
db = firebase.database()
auth = firebase.auth()

app = Flask(__name__)

class RepetitionCounter:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()
        self.voice_count = 0

    def voice_feedback(self):
        self.lock.acquire()
        self.engine.setProperty('volume', 2.0)
        self.engine.setProperty('rate', 130)
        self.engine.say("¡Bien hecho!, ¡Sigue así!, ¡No te detengas!, ¡Continúa haciendo el ejercicio!, ¡No te rindas!, ¡Excelente!")
        self.engine.runAndWait()
        self.lock.release()

    def count_repetitions_brazo(self, max_repetitions, camera_index=None):
        # Código para contar las repeticiones de brazos
        if camera_index is None:
            # Intenta abrir la cámara predeterminada
            cap = cv2.VideoCapture(0)
        else:
            # Intenta abrir la cámara especificada por el índice
            cap = cv2.VideoCapture(camera_index)
            
        if not cap.isOpened():
            print("No se pudo abrir la cámara.")
            return -1
        
        
        up = False
        down = False
        count = 0

        with self.mp_pose.Pose(static_image_mode=False) as pose:
            while True:
                ret, frame = cap.read()
                if ret is False:
                    break
                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks is not None:
                    x1 = int(results.pose_landmarks.landmark[11].x * width)
                    y1 = int(results.pose_landmarks.landmark[11].y * height)

                    x2 = int(results.pose_landmarks.landmark[13].x * width)
                    y2 = int(results.pose_landmarks.landmark[13].y * height)

                    x3 = int(results.pose_landmarks.landmark[15].x * width)
                    y3 = int(results.pose_landmarks.landmark[15].y * height)

                    p1 = np.array([x1,y1])
                    p2 = np.array([x2,y2])
                    p3 = np.array([x3,y3])

                    l1 = np.linalg.norm(p2-p3)
                    l2 = np.linalg.norm(p1-p3)
                    l3 = np.linalg.norm(p1-p2)

                    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 *l1 *l3)))
                    if angle >= 90:
                        up = True
                    if up == True and down == False and angle <= 40:
                        down = True
                    if up == True and down == True and angle >= 90:
                        count += 1
                        up = False
                        down = False
                        
                        # Incrementar el contador de voz
                        self.voice_count += 1
                        
                        # Verificar si se debe reproducir la voz
                        if self.voice_count == 4:
                            self.voice_count = 0  # Reiniciar el contador de voz
                            voice_thread = threading.Thread(target=self.voice_feedback)
                            voice_thread.start()
                        
                    if count >= max_repetitions: 
                        break

                    aux_image = np.zeros(frame.shape, np.uint8)
                    cv2.line(aux_image, (x1, y1), (x2, y2), (0, 100, 0), 20)
                    cv2.line(aux_image, (x2, y2), (x3, y3), (0, 100, 0), 20)
                    cv2.line(aux_image, (x1, y1), (x3, y3), (0, 100,), 5)
                    contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                    cv2.fillPoly(aux_image, pts=[contours], color=(128, 129, 129))
                    output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)

                    cv2.circle(output, (x1, y1), 6, (0, 0, 0), 4)
                    cv2.circle(output, (x2, y2), 6, (0, 0, 0), 4)
                    cv2.circle(output, (x3, y3), 6, (0, 0, 0), 4)
                    cv2.rectangle(output, (0, 0), (60, 60), (145,169, 115), -1)
                    cv2.putText(output, str(int(angle)), (x2 + 30, y2), 1, 1.5, (145,169, 115), 2)
                    cv2.putText(output, str(count), (10, 50), 1, 3.5, (0, 0, 0), 2)
                    cv2.imshow("output", output)
                
                if cv2.waitKey(1) & 0xFF ==27:
                    break
        
        # Liberar la captura de video y cerrar las ventanas de OpenCV
        cap.release()
        cv2.destroyAllWindows()
        
        return count

    def count_repetitions_pierna(self, max_repetitions):
        # Código para contar las repeticiones de piernas
        cap = cv2.VideoCapture(0)
        
        up = False
        down = False
        count = 0

        with self.mp_pose.Pose(static_image_mode=False) as pose:
            while True:
                ret, frame = cap.read()
                if ret is False:
                    break
                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks is not None:
                    x1 = int(results.pose_landmarks.landmark[23].x * width)
                    y1 = int(results.pose_landmarks.landmark[23].y * height)

                    x2 = int(results.pose_landmarks.landmark[25].x * width)
                    y2 = int(results.pose_landmarks.landmark[25].y * height)

                    x3 = int(results.pose_landmarks.landmark[27].x * width)
                    y3 = int(results.pose_landmarks.landmark[27].y * height)

                    p1 = np.array([x1,y1])
                    p2 = np.array([x2,y2])
                    p3 = np.array([x3,y3])

                    l1 = np.linalg.norm(p2-p3)
                    l2 = np.linalg.norm(p1-p3)
                    l3 = np.linalg.norm(p1-p2)

                    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 *l1 *l3)))
                    if angle >= 90:
                        up = True
                    if up == True and down == False and angle <= 40:
                        down = True
                    if up == True and down == True and angle >= 90:
                        count += 1
                        up = False
                        down = False
                        
                        # Incrementar el contador de voz
                        self.voice_count += 1
                        
                        # Verificar si se debe reproducir la voz
                        if self.voice_count == 4:
                            self.voice_count = 0  # Reiniciar el contador de voz
                            voice_thread = threading.Thread(target=self.voice_feedback)
                            voice_thread.start()
                        
                    if count >= max_repetitions: 
                        break

                    aux_image = np.zeros(frame.shape, np.uint8)
                    cv2.line(aux_image, (x1, y1), (x2, y2), (0, 100, 0), 20)
                    cv2.line(aux_image, (x2, y2), (x3, y3), (0, 100, 0), 20)
                    cv2.line(aux_image, (x1, y1), (x3, y3), (0, 100,), 5)
                    contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                    cv2.fillPoly(aux_image, pts=[contours], color=(128, 129, 129))
                    output = cv2.addWeighted(frame, 1, aux_image, 0.8, 0)

                    cv2.circle(output, (x1, y1), 6, (0, 0, 0), 4)
                    cv2.circle(output, (x2, y2), 6, (0, 0, 0), 4)
                    cv2.circle(output, (x3, y3), 6, (0, 0, 0), 4)
                    cv2.rectangle(output, (0, 0), (60, 60), (145,169, 115), -1)
                    cv2.putText(output, str(int(angle)), (x2 + 30, y2), 1, 1.5, (145,169, 115), 2)
                    cv2.putText(output, str(count), (10, 50), 1, 3.5, (0, 0, 0), 2)
                    cv2.imshow("output", output)
                
                if cv2.waitKey(1) & 0xFF ==27:
                    break
        
        # Liberar la captura de video y cerrar las ventanas de OpenCV
        cap.release()
        cv2.destroyAllWindows()
        
        return count

repetition_counter = RepetitionCounter()

def obtener_usuario_actual():
    try:
        # Verificar si hay un usuario autenticado en la sesión de Firebase
        usuario = auth.get_account_info(request.cookies.get('firebase_id_token'))
        return usuario
    except Exception as e:
        print("Error al obtener usuario actual:", str(e))
        return None

# Ruta para la página de inicio
@app.route('/')
def index():
    return render_template('ventanaInicio.html')

# Ruta para la página de rutinas
@app.route('/rutinas', methods=['GET', 'POST'])
def rutinas():
    usuario = obtener_usuario_actual()
    if usuario:
        usuario_id = usuario['users'][0]['localId']
        usuario_data = db.child("usuarios").child(usuario_id).get().val()
        return render_template('rutinas.html', usuario=usuario_data)
    else:
        return redirect(url_for('login'))

# Ruta para la página de login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        correo = request.form.get('correo')
        contrasena = request.form.get('contrasena')
        try:
            user = auth.sign_in_with_email_and_password(correo, contrasena)
            response = redirect(url_for('rutinas'))
            response.set_cookie('firebase_id_token', user['idToken'])
            return response
        except requests.exceptions.HTTPError as e:
            if "INVALID_PASSWORD" in str(e):
                error_message = "Contraseña incorrecta. Por favor, inténtalo de nuevo."
            elif "EMAIL_NOT_FOUND" in str(e):
                error_message = "Correo electrónico no encontrado. Por favor, regístrate primero."
            else:
                error_message = "Error al iniciar sesión. Por favor, inténtalo de nuevo."
            return render_template('login.html', error=error_message)
    return render_template('login.html')

# Ruta para la página de registro
@app.route('/registro', methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        # Obtener los datos del formulario
        nombre = request.form['nombre']
        apellido = request.form['apellido']
        fecha_nacimiento = request.form['fecha_nacimiento']
        edad = request.form['edad']
        peso = request.form['peso']
        altura = request.form['altura']
        correo = request.form['correo']
        contrasena = request.form['contrasena']

        # Crear el usuario en Firebase Authentication
        try:
            user = auth.create_user_with_email_and_password(correo, contrasena)
        except requests.exceptions.HTTPError as e:
            # Manejar errores de autenticación de Firebase
            error_message = e.args[1]
            return render_template('error.html', message=error_message)
        except Exception as e:
            # Manejar otros errores
            error_message = str(e)
            return render_template('error.html', message=error_message)

        # Guardar los datos del usuario en la base de datos de Firebase
        user_data = {
            "Nombre": nombre,
            "Apellido": apellido,
            "Fecha_Nacimiento": fecha_nacimiento,
            "Edad": edad,
            "Peso": peso,
            "Altura": altura,
            "Correo": correo
        }
        db.child("usuarios").child(user['localId']).set(user_data)

        # Redirigir al usuario a la página de inicio
        return redirect(url_for('index'))

    return render_template('registro.html')

#Redirigir al usuario a la página de información Brazos
@app.route('/informacionB')
def informacionB():
    return render_template('informacionB.html')

#Redirigir al usuario a la página de información Piernas 
@app.route('/informacionP')
def informacionP():
    return render_template('informacionP.html')

#Redirigir al usuario a la página de video Brazos
@app.route('/videoBrazos')
def videoB():
    return render_template('videoBrazos.html')

#Redirigir al usuario a la página de video Piernas
@app.route('/videoPiernas')
def videoP():
    return render_template('videoPiernas.html')

#Redirigir al usuario a la página de deteccion Brazos
@app.route('/start_detection_brazo', methods=['POST'])
def start_detection_brazo():
    max_repetitions = int(request.form['max_repetitions'])
    count = repetition_counter.count_repetitions_brazo(max_repetitions)
    
    # Obtener el usuario actual
    usuario = obtener_usuario_actual()
    if usuario:
        usuario_id = usuario['users'][0]['localId']
        
        # Crear un diccionario con la información de la repetición
        repeticion_data = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "numero_repeticiones": count
        }
        
# Obtener el número actual de repeticiones del usuario
        num_repeticiones = db.child("usuarios").child(usuario_id).child("repeticiones_brazos").shallow().get().val()
        if num_repeticiones is None:
            num_repeticiones = 0
        else:
            num_repeticiones = len(num_repeticiones)

        # Agregar la nueva repetición como un nuevo campo
        db.child("usuarios").child(usuario_id).child("repeticiones_brazos").child(str(num_repeticiones + 1)).set(repeticion_data)
        
    # Redirigir al usuario a la ventana de rutinas
        return redirect(url_for('rutinas'))
    else:
        return "No se pudo guardar la información. Usuario no autenticado."

# Ruta para mostrar los datos de las repeticiones
@app.route('/estadisticaBrazo')
def mostrar_datos_repeticiones():
    # Obtener el usuario actual
    usuario = obtener_usuario_actual()
    if usuario:
        usuario_id = usuario['users'][0]['localId']
        
        # Obtener los datos de las repeticiones del usuario
        repeticiones = db.child("usuarios").child(usuario_id).child("repeticiones_brazos").get().val()
        if repeticiones:
            return render_template('estadisticaBrazo.html', repeticiones=repeticiones)
        else:
            return "No hay datos de repeticiones para mostrar."
    else:
        return "Usuario no autenticado."


#Redirigir al usuario a la página de deteccion Piernas
@app.route('/start_detection_pierna', methods=['POST'])
def start_detection_pierna():
    max_repetitions = int(request.form['max_repetitions'])
    count = repetition_counter.count_repetitions_pierna(max_repetitions)
    
    # Obtener el usuario actual
    usuario = obtener_usuario_actual()
    if usuario:
        usuario_id = usuario['users'][0]['localId']
        
        # Crear un diccionario con la información de la repetición
        repeticion_data = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "numero_repeticiones": count
        }
        
# Obtener el número actual de repeticiones del usuario
        num_repeticiones = db.child("usuarios").child(usuario_id).child("repeticiones_piernas").shallow().get().val()
        if num_repeticiones is None:
            num_repeticiones = 0
        else:
            num_repeticiones = len(num_repeticiones)

        # Agregar la nueva repetición como un nuevo campo
        db.child("usuarios").child(usuario_id).child("repeticiones_piernas").child(str(num_repeticiones + 1)).set(repeticion_data)
        
    # Redirigir al usuario a la ventana de rutinas
        return redirect(url_for('rutinas'))
    else:
        return "No se pudo guardar la información. Usuario no autenticado."

# Ruta para mostrar los datos de las repeticiones
@app.route('/estadisticaPierna')
def mostrar_datos_repeticionesPiernas():
    # Obtener el usuario actual
    usuario = obtener_usuario_actual()
    if usuario:
        usuario_id = usuario['users'][0]['localId']
        
        # Obtener los datos de las repeticiones del usuario
        repeticiones = db.child("usuarios").child(usuario_id).child("repeticiones_piernas").get().val()
        if repeticiones:
            return render_template('estadisticaPierna.html', repeticiones=repeticiones)
        else:
            return "No hay datos de repeticiones para mostrar."
    else:
        return "Usuario no autenticado."


if __name__ == '__main__':
    app.run(host='192.168.18.14', port=8080, debug=True)

