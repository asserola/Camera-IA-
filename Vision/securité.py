import cv2
import numpy as np
import os
import pickle
import sqlite3
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk



class FaceDetector:
    def __init__(self):
        # Charger les classificateurs Haar Cascade pour la détection faciale
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Initialiser le recognizer LBPH pour la reconnaissance
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Stockage des visages et labels
        self.faces = []
        self.labels = []
        self.face_info = {}  # {label: {'id': '', 'name': ''}}

    def detect_faces(self, image):
        """Détecte les visages dans une image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Détection des visages avec paramètres optimisés
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return faces, gray

    def preprocess_face(self, face_roi):
        """Prétraite le visage pour la reconnaissance"""
        # Redimensionner à une taille standard
        face_roi = cv2.resize(face_roi, (200, 200))

        # Égalisation de l'histogramme pour améliorer le contraste
        face_roi = cv2.equalizeHist(face_roi)

        # Flou gaussien léger pour réduire le bruit
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)

        return face_roi

    def add_face(self, image, face_id, name):
        """Ajoute un visage à l'entraînement"""
        faces, gray = self.detect_faces(image)

        if len(faces) == 0:
            return False, "Aucun visage détecté"

        # Utiliser le premier visage détecté
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y + h, x:x + w]

        # Prétraiter le visage
        face_roi = self.preprocess_face(face_roi)

        # Générer un label unique
        label = len(self.faces) + 1

        # Ajouter aux données d'entraînement
        self.faces.append(face_roi)
        self.labels.append(label)

        # Stocker les informations du visage
        self.face_info[label] = {
            'id': face_id,
            'name': name,
            'added_date': datetime.now()
        }

        return True, f"Visage ajouté avec label {label}"

    def train_model(self):
        """Entraîne le modèle de reconnaissance"""
        if len(self.faces) > 0:
            self.face_recognizer.train(self.faces, np.array(self.labels))
            return True
        return False

    def recognize_face(self, image):
        """Reconnaît les visages dans une image"""
        faces, gray = self.detect_faces(image)
        results = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = self.preprocess_face(face_roi)

            if len(self.faces) > 0:
                # Prédire le visage
                label, confidence = self.face_recognizer.predict(face_roi)

                # Convertir la confiance (plus bas = meilleur)
                confidence_score = max(0, 100 - confidence) / 100

                if confidence_score > 0.5:  # Seuil de confiance
                    face_data = self.face_info.get(label, {})
                    face_id = face_data.get('id', f"unknown_{label}")
                    name = face_data.get('name', "Inconnu")
                else:
                    face_id = f"unknown_{datetime.now().strftime('%H%M%S')}"
                    name = "Inconnu"
                    confidence_score = 0.0
            else:
                face_id = f"unknown_{datetime.now().strftime('%H%M%S')}"
                name = "Inconnu"
                confidence_score = 0.0

            results.append({
                'face_id': face_id,
                'name': name,
                'confidence': confidence_score,
                'location': (x, y, w, h)
            })

        return results


class FaceRecognitionSystem:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.database_file = "face_database.db"
        self.model_file = "face_model.yml"
        self.setup_database()
        self.load_trained_data()

    def setup_database(self):
        """Initialise la base de données SQLite"""
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS faces
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           face_id
                           TEXT
                           UNIQUE,
                           name
                           TEXT,
                           label
                           INTEGER,
                           created_date
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           last_seen
                           TIMESTAMP
                       )
                       ''')
        conn.commit()
        conn.close()

    def load_trained_data(self):
        """Charge les données d'entraînement sauvegardées"""
        # Charger le modèle s'il existe
        if os.path.exists(self.model_file):
            try:
                self.face_detector.face_recognizer.read(self.model_file)

                # Charger les informations des visages
                if os.path.exists("face_data.pkl"):
                    with open("face_data.pkl", 'rb') as f:
                        data = pickle.load(f)
                        self.face_detector.faces = data['faces']
                        self.face_detector.labels = data['labels']
                        self.face_detector.face_info = data['face_info']

                print(f"Modèle chargé: {len(self.face_detector.faces)} visages")
            except Exception as e:
                print(f"Erreur lors du chargement: {e}")

    def save_trained_data(self):
        """Sauvegarde les données d'entraînement"""
        # Sauvegarder le modèle
        if len(self.face_detector.faces) > 0:
            self.face_detector.face_recognizer.write(self.model_file)

            # Sauvegarder les données supplémentaires
            data = {
                'faces': self.face_detector.faces,
                'labels': self.face_detector.labels,
                'face_info': self.face_detector.face_info
            }
            with open("face_data.pkl", 'wb') as f:
                pickle.dump(data, f)

    def add_face(self, image, name, face_id=None):
        """Ajoute un nouveau visage"""
        if face_id is None:
            face_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        success, message = self.face_detector.add_face(image, face_id, name)

        if success:
            # Entraîner le modèle
            self.face_detector.train_model()

            # Sauvegarder dans la base de données
            conn = sqlite3.connect(self.database_file)
            cursor = conn.cursor()

            # Trouver le label correspondant
            label = None
            for lbl, info in self.face_detector.face_info.items():
                if info['id'] == face_id:
                    label = lbl
                    break

            cursor.execute('''
                INSERT OR REPLACE INTO faces (face_id, name, label, last_seen)
                VALUES (?, ?, ?, ?)
            ''', (face_id, name, label, datetime.now()))

            conn.commit()
            conn.close()

            # Sauvegarder les données d'entraînement
            self.save_trained_data()

        return success, message

    def recognize_faces(self, image):
        """Reconnaît les visages dans une image"""
        return self.face_detector.recognize_face(image)

    def update_last_seen(self, face_id):
        """Met à jour la date de dernière vue"""
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE faces SET last_seen = ? WHERE face_id = ?",
            (datetime.now(), face_id)
        )
        conn.commit()
        conn.close()

    def get_all_faces(self):
        """Récupère tous les visages enregistrés"""
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()
        cursor.execute("SELECT face_id, name, created_date, last_seen FROM faces")
        faces = cursor.fetchall()
        conn.close()
        return faces

    def delete_face(self, face_id):
        """Supprime un visage"""
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()

        # Récupérer le label
        cursor.execute("SELECT label FROM faces WHERE face_id = ?", (face_id,))
        result = cursor.fetchone()

        if result:
            label = result[0]

            # Supprimer de la base de données
            cursor.execute("DELETE FROM faces WHERE face_id = ?", (face_id,))
            conn.commit()

            # Supprimer des données d'entraînement
            if label in self.face_detector.face_info:
                del self.face_detector.face_info[label]

            # Reconstruire les listes faces et labels
            new_faces = []
            new_labels = []
            new_face_info = {}
            new_label_map = {}

            current_label = 1
            for old_label, info in self.face_detector.face_info.items():
                new_face_info[current_label] = info
                new_label_map[old_label] = current_label
                current_label += 1

            # Reconstruire faces et labels
            for i, old_label in enumerate(self.face_detector.labels):
                if old_label in new_label_map:
                    new_faces.append(self.face_detector.faces[i])
                    new_labels.append(new_label_map[old_label])

            self.face_detector.faces = new_faces
            self.face_detector.labels = new_labels
            self.face_detector.face_info = new_face_info

            # Réentraîner le modèle
            if len(new_faces) > 0:
                self.face_detector.train_model()
                self.save_trained_data()
            else:
                # Supprimer les fichiers si plus de visages
                if os.path.exists(self.model_file):
                    os.remove(self.model_file)
                if os.path.exists("face_data.pkl"):
                    os.remove("face_data.pkl")

        conn.close()
        return True


class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Système de Reconnaissance Faciale - OpenCV")
        self.root.geometry("1200x700")

        self.face_system = FaceRecognitionSystem()
        self.camera_active = False
        self.cap = None

        self.setup_gui()

    def setup_gui(self):
        """Configure l'interface graphique"""
        # Style
        style = ttk.Style()
        style.configure('TButton', padding=5, font=('Arial', 10))

        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame gauche (caméra)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Label pour l'affichage de la caméra
        self.camera_label = ttk.Label(left_frame, background='black')
        self.camera_label.pack(pady=5, fill=tk.BOTH, expand=True)

        # Frame des boutons de contrôle
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(pady=10)

        ttk.Button(control_frame, text="Démarrer Caméra",
                   command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Arrêter Caméra",
                   command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Ajouter Visage",
                   command=self.add_face_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Capturer Image",
                   command=self.capture_image).pack(side=tk.LEFT, padx=5)

        # Frame droite (informations)
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)

        # Informations système
        info_frame = ttk.LabelFrame(right_frame, text="Informations Système", padding=10)
        info_frame.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(info_frame, text="Prêt")
        self.status_label.pack()

        ttk.Label(info_frame, text=f"Visages enregistrés: {len(self.face_system.face_detector.faces)}").pack()

        # Liste des visages
        list_frame = ttk.LabelFrame(right_frame, text="Visages Enregistrés", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Treeview pour afficher les visages
        columns = ('ID', 'Nom', 'Date')
        self.faces_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=12)

        for col in columns:
            self.faces_tree.heading(col, text=col)
            self.faces_tree.column(col, width=80)

        self.faces_tree.pack(fill=tk.BOTH, expand=True)

        # Barre de défilement
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.faces_tree.yview)
        self.faces_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Boutons de gestion
        manage_frame = ttk.Frame(list_frame)
        manage_frame.pack(pady=5)

        ttk.Button(manage_frame, text="Rafraîchir",
                   command=self.refresh_faces_list).pack(side=tk.LEFT, padx=2)
        ttk.Button(manage_frame, text="Supprimer",
                   command=self.delete_selected_face).pack(side=tk.LEFT, padx=2)

        # Détections en temps réel
        detection_frame = ttk.LabelFrame(right_frame, text="Détections", padding=10)
        detection_frame.pack(fill=tk.BOTH, pady=5)

        self.detection_text = tk.Text(detection_frame, height=8, width=35)
        self.detection_text.pack(fill=tk.BOTH)

        self.refresh_faces_list()

    def start_camera(self):
        """Démarre la caméra"""
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erreur", "Impossible d'accéder à la caméra")
                return

            self.camera_active = True
            self.status_label.config(text="Caméra active")
            self.update_camera()

    def stop_camera(self):
        """Arrête la caméra"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.camera_label.configure(image='')
        self.status_label.config(text="Caméra arrêtée")

    def update_camera(self):
        """Met à jour l'affichage de la caméra"""
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Redimensionner pour l'affichage
                display_frame = cv2.resize(frame, (640, 480))

                # Reconnaître les visages
                results = self.face_system.recognize_faces(display_frame)

                # Mettre à jour le texte des détections
                self.detection_text.delete(1.0, tk.END)
                if results:
                    for result in results:
                        self.detection_text.insert(tk.END,
                                                   f"ID: {result['face_id']}\n"
                                                   f"Nom: {result['name']}\n"
                                                   f"Confiance: {result['confidence']:.2%}\n"
                                                   f"--------------------\n"
                                                   )

                # Dessiner les rectangles
                for result in results:
                    x, y, w, h = result['location']
                    color = (0, 255, 0) if result['confidence'] > 0.5 else (0, 0, 255)

                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display_frame,
                                f"{result['name']} ({result['confidence']:.1%})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Convertir pour Tkinter
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)

                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

            self.root.after(30, self.update_camera)

    def add_face_dialog(self):
        """Ouvre une boîte de dialogue pour ajouter un visage"""
        if not self.camera_active:
            messagebox.showerror("Erreur", "Veuillez démarrer la caméra d'abord")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Ajouter un nouveau visage")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="ID du visage:").pack(pady=5)
        id_entry = ttk.Entry(dialog, width=30)
        id_entry.pack(pady=5)

        ttk.Label(dialog, text="Nom:").pack(pady=5)
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack(pady=5)

        def capture_and_add():
            face_id = id_entry.get().strip()
            name = name_entry.get().strip()

            if not face_id or not name:
                messagebox.showerror("Erreur", "Veuillez remplir tous les champs")
                return

            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    success, message = self.face_system.add_face(frame, name, face_id)
                    if success:
                        messagebox.showinfo("Succès", message)
                        self.refresh_faces_list()
                        dialog.destroy()
                    else:
                        messagebox.showerror("Erreur", message)

        ttk.Button(dialog, text="Capturer et Ajouter",
                   command=capture_and_add).pack(pady=10)

    def capture_image(self):
        """Capture une image depuis la caméra"""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                messagebox.showinfo("Succès", f"Image sauvegardée: {filename}")

    def refresh_faces_list(self):
        """Rafraîchit la liste des visages"""
        for item in self.faces_tree.get_children():
            self.faces_tree.delete(item)

        faces = self.face_system.get_all_faces()
        for face_id, name, created, last_seen in faces:
            self.faces_tree.insert('', tk.END, values=(face_id, name, created[:10]))

    def delete_selected_face(self):
        """Supprime le visage sélectionné"""
        selection = self.faces_tree.selection()
        if selection:
            item = selection[0]
            face_id = self.faces_tree.item(item, 'values')[0]
            name = self.faces_tree.item(item, 'values')[1]

            if messagebox.askyesno("Confirmation",
                                   f"Voulez-vous vraiment supprimer {name} ({face_id})?"):
                if self.face_system.delete_face(face_id):
                    messagebox.showinfo("Succès", "Visage supprimé")
                    self.refresh_faces_list()

    def __del__(self):
        """Nettoyage à la fermeture"""
        self.stop_camera()


def main():
    # Vérifier si OpenCV est installé
    try:
        import cv2
        from PIL import Image, ImageTk
    except ImportError as e:
        print(f"Erreur: Dépendances manquantes. Installez avec:")
        print("pip install opencv-python pillow")
        return

    root = tk.Tk()
    app = FaceRecognitionGUI(root)

    # Gérer la fermeture propre
    def on_closing():
        app.stop_camera()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()