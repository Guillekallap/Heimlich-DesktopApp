# src/main.py
import sys
import cv2
import base64, requests, json
import resources_rc
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice
from pathlib import Path
from PySide6.QtCore import Qt


def load_ui(path):
    loader = QUiLoader()
    ui_file = QFile(str(path))
    ui_file.open(QIODevice.ReadOnly)
    window = loader.load(ui_file)
    ui_file.close()
    return window


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        base_dir = Path(__file__).resolve().parents[1]
        self.ui = load_ui(base_dir / "ui" / "main_window.ui")

        # Buscar el QLabel del .ui
        self.label: QLabel = self.ui.findChild(QLabel, "lblCamera")

        # Qlabel que muestra el resultado de cada captura
        self.lbl_result: QLabel = self.ui.findChild(QLabel, "lblResult")

        # Imagen correcta
        self.correcto_icon = QPixmap(":/icons/correcto.png")
        self.correcto_icon = self.correcto_icon.scaled(self.lbl_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_result.setPixmap(self.correcto_icon)

        #imagen error
        self.error_icon = QPixmap(":/icons/error.png")
        self.error_icon = self.error_icon.scaled(self.lbl_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Inicializar la cámara (0 = webcam principal)
        self.cap = cv2.VideoCapture(0)

        # Crear un QTimer que actualice el frame cada 30 ms
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Timer de captura cada 3 segundos
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_to_base64)
        self.capture_timer.start(3000)

        self.ui.show()

    def post_request(self,image: str):
        payload = {"image": image}  # se pasa una imagen al request
        print("Enviando request al servidor.")
        try:
            r = requests.post("http://127.0.0.1:8000/predictOne", json=payload, timeout=120)

        except requests.RequestException as e:
            # Error de red / timeout
            print(f"[POST] Error de red: {e}")
            return {"error": str(e)}

        try:
            data = r.json()

        except ValueError:
            # No era JSON
            return {"error": "Respuesta no es JSON", "raw": r.text}

        if not r.ok:
            # FastAPI suele usar {"detail": ...}
            return {"error": "HTTP error", "status": r.status_code, "detail": data.get("detail", data)}

        prediction = data["prediction"]

        if prediction is None:
            # No vino 'prediction'; devuelve lo que vino para depurar
            return {"error": "Campo 'prediction' ausente", "data": data}

        print("Resultado de prediccion:",prediction)
        return prediction



    def set_icon_result(self,result: str):
        if result == "correcta":
            self.lbl_result.setPixmap(self.correcto_icon)
        else:
            self.lbl_result.setPixmap(self.error_icon)

    def capture_to_base64(self):
        """Toma el último frame y lo convierte a base64 (JPEG)."""
        if self.last_frame_bgr is None:
            return

        # Codificar a JPEG en memoria (calidad 90)
        ok, buf = cv2.imencode(".jpg", self.last_frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            print("No se pudo codificar el frame.")
            return

        print("Captura de foto.")

        b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")

        result = self.post_request(b64_str)

        if "error" in result:
            print(f"[PREDICT] Error: {result['error']}")
            if "status" in result:
                print(f"[PREDICT] HTTP status: {result['status']}")
            if "detail" in result:
                print(f"[PREDICT] detail: {result['detail']}")
            # opcional: mostrar un ícono de error en la UI
            return

        pred = result["prediction"]
        avg = result.get("average")
        print(f"[PREDICT] prediction={pred}, average={avg}")

        # agrega el icono de correcto o incorrecto
        self.set_icon_result(result)



    def update_frame(self):
        ok, frame_bgr = self.cap.read()
        if not ok:
            return

        self.last_frame_bgr = frame_bgr  # ¡guardar último frame!

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        """Cerrar cámara al cerrar la app"""
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    sys.exit(app.exec())
