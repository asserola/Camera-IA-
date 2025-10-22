import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import face_recognition
import tkinter as tk
from tkinter import messagebox, filedialog
import sqlite3
from PIL import Image, ImageTk
import threading

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.database_file = "face_database.db"
        self.encodings_file = "face_encodings.pkl"
        self.setup_database()
        self.load_known_faces()
        
    def setup_database(self):
        """Initialise la base de données SQLite"""
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT UNIQUE,
                name TEXT,
                encoding BLOB,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def load_known_faces(self):
        """Charge les visages connus depuis la base de données"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_ids = data['ids']
                    self.known_face_names = data['names']
                print(f"Chargé {len(self.known_face_encodings)} visages connus")
            except Exception as e:
                print(f"Erreur lors du chargement: {e}")
        
        # Synchroniser avec la base de données
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()
        cursor.execute("SELECT face_id, name FROM faces")
        rows = cursor.fetchall()
        conn.close()
    
    def save_known_faces(self):
        """Sauvegarde les encodages des visages"""
        data = {
            'encodings': self.known_face_encodings,
            'ids': self.known_face_ids,
            'names': self.known_face_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def add_face(self, image, name, face_id=None):
        """Ajoute un nouveau visage à la base de données"""
        if face_id is None:
            face_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Détecter et encoder le visage
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_encodings) == 0:
            return False, "Aucun visage détecté"
        
        # Utiliser le premier visage détecté
        face_encoding = face_encodings[0]
        
        # Vérifier si le face_id existe déjà
        if face_id in self.known_face_ids:
            return False, "ID déjà existant"
        
        # Ajouter aux listes en mémoire
        self.known_face_encodings.append(face_encoding)
        self.known_face_ids.append(face_id)
        self.known_face_names.append(name)
        
        # Sauvegarder dans la base de données
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO faces (face_id, name, encoding, last_seen)
            VALUES (?, ?, ?, ?)
        ''', (face_id, name, pickle.dumps(face_encoding), datetime.now()))
        conn.commit()
        conn.close()
        
        self.save_known_faces()
        return True, f"Visage ajouté avec ID: {face_id}"
    
    def recognize_faces(self, image):
        """Reconnaît les visages dans une image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Détecter les visages
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        results = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Comparer avec les visages connus
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=0.6
                )
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                
                best_match_index = np.argmin(face_distances) if face_distances.any() else -1
                
                if best_match_index != -1 and matches[best_match_index]:
                    face_id = self.known_face_ids[best_match_index]
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                else:
                    face_id = f"unknown_{datetime.now().strftime('%H%M%S')}"
                    name = "Inconnu"
                    confidence = 0.0
            else:
                face_id = f"unknown_{datetime.now().strftime('%H%M%S')}"
                name = "Inconnu"
                confidence = 0.0
            
            top, right, bottom, left = face_location
            results.append({
                'face_id': face_id,
                'name': name,
                'confidence': confidence,
                'location': (left, top, right, bottom)
            })
            
            # Mettre à jour le last_seen dans la base de données
            if name != "Inconnu":
                self.update_last_seen(face_id)
        
        return results
    
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
        """Supprime un visage de la base de données"""
        if face_id in self.known_face_ids:
            index = self.known_face_ids.index(face_id)
            self.known_face_encodings.pop(index)
            self.known_face_ids.pop(index)
            self.known_face_names.pop(index)
            
            conn = sqlite3.connect(self.database_file)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM faces WHERE face_id = ?", (face_id,))
            conn.commit()
            conn.close()
            
            self.save_known_faces()
            return True
        return False

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Système de Reconnaissance Faciale Professionnel")
        self.root.geometry("1200x700")
        
        self.face_system = FaceRecognitionSystem()
        self.camera_active = False
        self.cap = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Configure l'interface graphique"""
        # Frame principale
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame gauche (camera et contrôles)
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Label pour l'affichage de la caméra
        self.camera_label = tk.Label(left_frame, bg='black', width=80, height=25)
        self.camera_label.pack(pady=5)
        
        # Frame des boutons de contrôle
        control_frame = tk.Frame(left_frame)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="Démarrer Caméra", 
                 command=self.start_camera, bg='green', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Arrêter Caméra", 
                 command=self.stop_camera, bg='red', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Ajouter Visage", 
                 command=self.add_face_dialog, bg='blue', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Capturer Image", 
                 command=self.capture_image, bg='orange', fg='white').pack(side=tk.LEFT, padx=5)
        
        # Frame droite (liste des visages et informations)
        right_frame = tk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)
        
        # Liste des visages enregistrés
        tk.Label(right_frame, text="Visages Enregistrés", 
                font=('Arial', 12, 'bold')).pack(pady=5)
        
        self.faces_listbox = tk.Listbox(right_frame, height=15)
        self.faces_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.faces_listbox.bind('<<ListboxSelect>>', self.on_face_select)
        
        # Boutons de gestion
        manage_frame = tk.Frame(right_frame)
        manage_frame.pack(pady=5)
        
        tk.Button(manage_frame, text="Rafraîchir", 
                 command=self.refresh_faces_list).pack(side=tk.LEFT, padx=2)
        tk.Button(manage_frame, text="Supprimer", 
                 command=self.delete_selected_face, bg='red', fg='white').pack(side=tk.LEFT, padx=2)
        
        # Informations du visage sélectionné
        self.info_text = tk.Text(right_frame, height=8, width=35)
        self.info_text.pack(fill=tk.BOTH, pady=5)
        
        self.refresh_faces_list()
    
    def start_camera(self):
        """Démarre la caméra"""
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            self.camera_active = True
            self.update_camera()
    
    def stop_camera(self):
        """Arrête la caméra"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.camera_label.configure(image='')
    
    def update_camera(self):
        """Met à jour l'affichage de la caméra"""
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Redimensionner le frame pour l'affichage
                frame = cv2.resize(frame, (640, 480))
                
                # Reconnaître les visages
                results = self.face_system.recognize_faces(frame)
                
                # Dessiner les rectangles et informations
                for result in results:
                    left, top, right, bottom = result['location']
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    label = f"{result['name']} ({result['face_id']})"
                    confidence = f"{result['confidence']:.2%}"
                    
                    cv2.putText(frame, label, (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, confidence, (left, bottom + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Convertir pour l'affichage Tkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            
            self.root.after(10, self.update_camera)
    
    def add_face_dialog(self):
        """Ouvre une boîte de dialogue pour ajouter un visage"""
        if not self.camera_active:
            messagebox.showerror("Erreur", "Veuillez démarrer la caméra d'abord")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Ajouter un nouveau visage")
        dialog.geometry("300x200")
        
        tk.Label(dialog, text="ID du visage:").pack(pady=5)
        id_entry = tk.Entry(dialog, width=30)
        id_entry.pack(pady=5)
        
        tk.Label(dialog, text="Nom:").pack(pady=5)
        name_entry = tk.Entry(dialog, width=30)
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
        
        tk.Button(dialog, text="Capturer et Ajouter", 
                 command=capture_and_add, bg='green', fg='white').pack(pady=10)
    
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
        self.faces_listbox.delete(0, tk.END)
        faces = self.face_system.get_all_faces()
        for face_id, name, created, last_seen in faces:
            self.faces_listbox.insert(tk.END, f"{name} ({face_id})")
    
    def on_face_select(self, event):
        """Affiche les informations du visage sélectionné"""
        selection = self.faces_listbox.curselection()
        if selection:
            index = selection[0]
            faces = self.face_system.get_all_faces()
            if index < len(faces):
                face_id, name, created, last_seen = faces[index]
                
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, f"ID: {face_id}\n")
                self.info_text.insert(tk.END, f"Nom: {name}\n")
                self.info_text.insert(tk.END, f"Créé le: {created}\n")
                self.info_text.insert(tk.END, f"Dernière vue: {last_seen if last_seen else 'Jamais'}\n")
    
    def delete_selected_face(self):
        """Supprime le visage sélectionné"""
        selection = self.faces_listbox.curselection()
        if selection:
            index = selection[0]
            faces = self.face_system.get_all_faces()
            if index < len(faces):
                face_id = faces[index][0]
                name = faces[index][1]
                
                if messagebox.askyesno("Confirmation", 
                                      f"Voulez-vous vraiment supprimer {name} ({face_id})?"):
                    if self.face_system.delete_face(face_id):
                        messagebox.showinfo("Succès", "Visage supprimé")
                        self.refresh_faces_list()
                        self.info_text.delete(1.0, tk.END)
    
    def __del__(self):
        """Nettoyage à la fermeture"""
        self.stop_camera()

def main():
    # Vérifier les dépendances
    try:
        import face_recognition
        import cv2
    except ImportError as e:
        print(f"Erreur: Dépendances manquantes. Installez avec:")
        print("pip install face-recognition opencv-python pillow")
        return
    
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()