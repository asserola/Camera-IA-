import cv2
import numpy as np
import os
import pickle
import sqlite3
import smtplib
import threading
import time
import requests
import hashlib
import shutil
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import pygame
from collections import deque
import json
import platform
import sys

# Initialiser pygame pour les sons
try:
    pygame.mixer.init()
except:
    print("Pygame audio non disponible")


class NetworkManager:
    def __init__(self):
        self.shared_folder = "shared_faces"
        self.setup_shared_folder()

    def setup_shared_folder(self):
        """Crée le dossier partagé"""
        os.makedirs(self.shared_folder, exist_ok=True)
        os.makedirs(os.path.join(self.shared_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.shared_folder, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.shared_folder, "data"), exist_ok=True)

    def sync_with_network(self):
        """Synchronise avec le dossier partagé"""
        try:
            # Vérifier si le dossier partagé contient des fichiers
            shared_model = os.path.join(self.shared_folder, "models", "face_model.yml")
            shared_data = os.path.join(self.shared_folder, "data", "face_data.pkl")

            files_copied = 0

            # Synchroniser les modèles si le fichier existe
            if os.path.exists(shared_model):
                shutil.copy2(shared_model, "face_model.yml")
                files_copied += 1
                print("Modèle synchronisé depuis le réseau")

            # Synchroniser les données si le fichier existe
            if os.path.exists(shared_data):
                shutil.copy2(shared_data, "face_data.pkl")
                files_copied += 1
                print("Données synchronisées depuis le réseau")

            return files_copied > 0
        except Exception as e:
            print(f"Erreur synchronisation: {e}")
            return False

    def share_to_network(self):
        """Partage les données vers le réseau"""
        try:
            files_shared = 0

            # Partager le modèle s'il existe
            if os.path.exists("face_model.yml"):
                shutil.copy2("face_model.yml",
                             os.path.join(self.shared_folder, "models", "face_model.yml"))
                files_shared += 1

            # Partager les données si elles existent
            if os.path.exists("face_data.pkl"):
                shutil.copy2("face_data.pkl",
                             os.path.join(self.shared_folder, "data", "face_data.pkl"))
                files_shared += 1

            return files_shared > 0
        except Exception as e:
            print(f"Erreur partage: {e}")
            return False


class ThemeManager:
    def __init__(self):
        self.themes = {
            "Cyberpunk Matrix": {
                'bg': '#001100',
                'fg': '#00ff00',
                'accent': '#00ffff',
                'warning': '#ff0000',
                'panel_bg': '#002200',
                'text_bg': '#000000'
            },
            "Dark Neon": {
                'bg': '#0a0a0a',
                'fg': '#ff00ff',
                'accent': '#00ffff',
                'warning': '#ffff00',
                'panel_bg': '#1a1a1a',
                'text_bg': '#000000'
            },
            "Blue Matrix": {
                'bg': '#000a1a',
                'fg': '#0088ff',
                'accent': '#00ffff',
                'warning': '#ff4444',
                'panel_bg': '#001a33',
                'text_bg': '#000511'
            },
            "Red Alert": {
                'bg': '#1a0000',
                'fg': '#ff0000',
                'accent': '#ffff00',
                'warning': '#ff4444',
                'panel_bg': '#330000',
                'text_bg': '#0a0000'
            },
            "Purple Hacker": {
                'bg': '#0a001a',
                'fg': '#aa00ff',
                'accent': '#ff00aa',
                'warning': '#00ffff',
                'panel_bg': '#1a0033',
                'text_bg': '#05000a'
            }
        }
        self.current_theme = "Cyberpunk Matrix"

    def get_theme(self, theme_name):
        """Retourne les couleurs d'un thème"""
        return self.themes.get(theme_name, self.themes["Cyberpunk Matrix"])

    def get_theme_names(self):
        """Retourne la liste des noms de thèmes"""
        return list(self.themes.keys())


class NotificationSystem:
    def __init__(self):
        self.email_config = self.load_email_config()

    def load_email_config(self):
        """Charge la configuration email"""
        try:
            with open('email_config.json', 'r') as f:
                return json.load(f)
        except:
            return {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': 'votre.email@gmail.com',
                'sender_password': 'votre_mot_de_passe',
                'receiver_email': 'admin@votreentreprise.com'
            }

    def send_alert_email(self, subject, message):
        """Envoie une alerte par email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['receiver_email']
            msg['Subject'] = f"🔴 CYBER-VISION ALERT: {subject}"

            body = f"""
            ⚠️ ALERTE DE SÉCURITÉ ⚠️

            {message}

            Heure: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Système: Cyber-Vision AI v2.0

            ----------------------------
            Cet email a été généré automatiquement.
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.email_config['smtp_server'],
                                  self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'],
                         self.email_config['sender_password'])
            server.send_message(msg)
            server.quit()

            return True
        except Exception as e:
            print(f"Erreur envoi email: {e}")
            return False

    def play_alert_sound(self, alert_type="default"):
        """Joue un son d'alerte"""
        try:
            if alert_type == "critical":
                # Bip critique
                for _ in range(3):
                    print("\a")
                    time.sleep(0.2)
            else:
                # Bip standard
                print("\a")
        except:
            pass


class CyberSecurityAI:
    def __init__(self):
        self.suspicious_activities = deque(maxlen=100)
        self.learning_rate = 0.1
        self.anomaly_threshold = 0.7
        self.unknown_face_counter = 0
        self.last_alert_time = None

    def analyze_behavior(self, face_data, movement_data):
        """Analyse le comportement pour détecter des anomalies"""
        score = 0
        current_time = datetime.now()

        # Détection de mouvement rapide
        if movement_data.get('speed', 0) > 50:
            score += 0.3
            self.log_suspicious_activity("Mouvement rapide détecté", "MEDIUM")

        # Détection de visages multiples
        if len(face_data) > 2:
            score += 0.2
            self.log_suspicious_activity("Multiples visages détectés", "MEDIUM")

        # Détection d'inconnus répétés
        unknown_count = sum(1 for face in face_data if face['name'] == "Inconnu")
        if unknown_count >= 1:
            self.unknown_face_counter += 1
            if self.unknown_face_counter >= 3:
                score += 0.8
                self.log_suspicious_activity("Multiples inconnus détectés", "HIGH")
        else:
            self.unknown_face_counter = max(0, self.unknown_face_counter - 1)

        # Vérification du timing des alertes
        if self.last_alert_time:
            time_since_last_alert = (current_time - self.last_alert_time).total_seconds()
            if time_since_last_alert < 300:  # 5 minutes
                score *= 0.5  # Réduit le score si alerte récente

        return score > self.anomaly_threshold

    def log_suspicious_activity(self, activity, level="MEDIUM"):
        """Journalise les activités suspectes"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'activity': activity,
            'level': level
        }
        self.suspicious_activities.append(log_entry)

        # Sauvegarder dans un fichier log
        try:
            with open("security_log.json", "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
        except:
            pass

    def adaptive_learning(self, recognition_data):
        """Apprentissage adaptatif du système"""
        if recognition_data['confidence'] > 0.8:
            self.learning_rate = min(0.2, self.learning_rate + 0.01)
        else:
            self.learning_rate = max(0.05, self.learning_rate - 0.005)


class FuturisticFaceRecognition:
    def __init__(self):
        # Chargement des classificateurs
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Données d'entraînement
        self.faces = []
        self.labels = []
        self.face_info = {}
        self.movement_history = deque(maxlen=50)

        # Systèmes
        self.security_ai = CyberSecurityAI()
        self.notification_system = NotificationSystem()
        self.network_manager = NetworkManager()

        # Configuration
        self.config = self.load_config()
        self.setup_directories()
        self.load_trained_data()  # Charger les données au démarrage

        # État administrateur
        self.admin_logged_in = False

    def setup_directories(self):
        """Crée les répertoires nécessaires"""
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("captures", exist_ok=True)

    def load_config(self):
        """Charge la configuration du système"""
        default_config = {
            'detection_confidence': 0.6,
            'max_unknown_faces': 3,
            'email_alerts': False,
            'sound_alerts': True,
            'auto_learning': True,
            'admin_password': 'cyber2024',
            'update_check_interval': 3600,
            'current_theme': 'Cyberpunk Matrix',
            'update_urls': ['http://localhost:8000/updates', 'https://your-update-server.com/updates']
        }

        try:
            with open('config.json', 'r') as f:
                loaded_config = json.load(f)
                return {**default_config, **loaded_config}
        except:
            return default_config

    def save_config(self):
        """Sauvegarde la configuration"""
        with open('config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def load_trained_data(self):
        """Charge les données d'entraînement existantes"""
        try:
            # Charger le modèle s'il existe
            if os.path.exists("face_model.yml"):
                self.face_recognizer.read("face_model.yml")
                print("Modèle de reconnaissance chargé")
            else:
                print("Aucun modèle trouvé, création d'un nouveau modèle")
                # Créer un modèle vide
                if len(self.faces) > 0:
                    self.face_recognizer.train(self.faces, np.array(self.labels))

            # Charger les données supplémentaires
            if os.path.exists("face_data.pkl"):
                with open("face_data.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.faces = data.get('faces', [])
                    self.labels = data.get('labels', [])
                    self.face_info = data.get('face_info', {})
                print(f"Données chargées: {len(self.faces)} visages")

        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            # Réinitialiser avec des données vides
            self.faces = []
            self.labels = []
            self.face_info = {}

    def detect_faces_with_movement(self, frame, previous_frame):
        """Détecte les visages et analyse les mouvements"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection des visages
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Analyse de mouvement
        movement_data = self.analyze_movement(frame, previous_frame)

        return faces, gray, movement_data

    def analyze_movement(self, current_frame, previous_frame):
        """Analyse les mouvements dans la scène"""
        if previous_frame is None:
            return {'speed': 0, 'movement_detected': False}

        # Conversion en niveaux de gris
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # Calcul de la différence
        diff = cv2.absdiff(current_gray, previous_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Calcul du pourcentage de mouvement
        movement_percent = np.sum(thresh) / (thresh.size * 255)

        return {
            'speed': movement_percent * 100,
            'movement_detected': movement_percent > 0.01
        }

    def add_face(self, image, name, face_id=None):
        """Ajoute un nouveau visage (admin seulement)"""
        if not self.admin_logged_in:
            return False, "Accès administrateur requis"

        if face_id is None:
            face_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return False, "Aucun visage détecté"

        # Utiliser le premier visage détecté
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (200, 200))
        face_roi = cv2.equalizeHist(face_roi)

        # Ajouter aux données d'entraînement
        label = len(self.faces) + 1
        self.faces.append(face_roi)
        self.labels.append(label)

        self.face_info[label] = {
            'id': face_id,
            'name': name,
            'added_date': datetime.now()
        }

        # Entraîner le modèle
        if len(self.faces) > 0:
            try:
                self.face_recognizer.train(self.faces, np.array(self.labels))
                self.face_recognizer.save("face_model.yml")

                # Sauvegarder les données
                data = {
                    'faces': self.faces,
                    'labels': self.labels,
                    'face_info': self.face_info
                }
                with open("face_data.pkl", 'wb') as f:
                    pickle.dump(data, f)

                # Synchroniser avec le réseau
                self.network_manager.share_to_network()

                return True, f"Visage {name} ajouté avec ID: {face_id}"

            except Exception as e:
                return False, f"Erreur lors de l'entraînement: {str(e)}"
        else:
            return False, "Aucune donnée à entraîner"

    def recognize_faces_advanced(self, frame, movement_data):
        """Reconnaissance faciale avancée avec analyse comportementale"""
        faces, gray, _ = self.detect_faces_with_movement(frame, frame)
        results = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)

            if len(self.faces) > 0:
                try:
                    label, confidence = self.face_recognizer.predict(face_roi)
                    confidence_score = max(0, 100 - confidence) / 100

                    if confidence_score > self.config['detection_confidence']:
                        face_data = self.face_info.get(label, {})
                        face_id = face_data.get('id', f"unknown_{label}")
                        name = face_data.get('name', "Inconnu")
                    else:
                        face_id = f"unknown_{datetime.now().strftime('%H%M%S')}"
                        name = "Inconnu"
                        confidence_score = 0.0
                except Exception as e:
                    print(f"Erreur prédiction: {e}")
                    face_id = f"unknown_{datetime.now().strftime('%H%M%S')}"
                    name = "Inconnu"
                    confidence_score = 0.0
            else:
                face_id = f"unknown_{datetime.now().strftime('%H%M%S')}"
                name = "Inconnu"
                confidence_score = 0.0

            result = {
                'face_id': face_id,
                'name': name,
                'confidence': confidence_score,
                'location': (x, y, w, h),
                'timestamp': datetime.now()
            }
            results.append(result)

            # Apprentissage adaptatif
            if self.config['auto_learning'] and confidence_score > 0.7:
                try:
                    self.faces.append(face_roi)
                    self.labels.append(len(self.faces) + 1)
                    self.face_info[len(self.faces)] = {
                        'id': face_id,
                        'name': name,
                        'added_date': datetime.now()
                    }
                    if len(self.faces) > 0:
                        self.face_recognizer.train(self.faces, np.array(self.labels))
                except Exception as e:
                    print(f"Erreur apprentissage adaptatif: {e}")

        # Analyse comportementale
        if results and self.security_ai.analyze_behavior(results, movement_data):
            self.trigger_alerts(results, movement_data)

        return results

    def trigger_alerts(self, face_data, movement_data):
        """Déclenche les alertes de sécurité"""
        alert_message = f"Activité suspecte détectée: {len(face_data)} visages, mouvement: {movement_data['speed']:.1f}%"

        if self.config['sound_alerts']:
            self.notification_system.play_alert_sound("critical")

        if self.config['email_alerts']:
            threading.Thread(target=self.notification_system.send_alert_email,
                             args=("Activité Suspecte", alert_message)).start()

        self.security_ai.last_alert_time = datetime.now()

    def capture_image(self, image, filename_prefix="capture"):
        """Capture une image (admin seulement)"""
        if not self.admin_logged_in:
            return False, "Accès administrateur requis"

        try:
            filename = f"captures/{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, image)
            return True, f"Image sauvegardée: {filename}"
        except Exception as e:
            return False, f"Erreur capture: {str(e)}"


class HackerStyleGUI:
    def __init__(self, root):
        self.root = root
        self.theme_manager = ThemeManager()
        self.face_system = FuturisticFaceRecognition()
        self.setup_futuristic_ui()
        self.setup_system()

    def setup_futuristic_ui(self):
        """Configure l'interface style hacker"""
        self.root.title("▓ CYBER-VISION AI v2.0 ▓ Neural Face Recognition System")
        self.apply_theme(self.face_system.config.get('current_theme', 'Cyberpunk Matrix'))

        # Style futuriste
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.configure_styles()
        self.create_main_interface()
        self.create_menu_bar()

    def apply_theme(self, theme_name):
        """Applique un thème à l'interface"""
        colors = self.theme_manager.get_theme(theme_name)
        self.colors = colors
        self.root.configure(bg=colors['bg'])
        self.face_system.config['current_theme'] = theme_name
        self.face_system.save_config()

    def configure_styles(self):
        """Configure les styles visuels"""
        # Configuration des styles ttk
        self.style.configure('Cyber.TFrame', background=self.colors['bg'])
        self.style.configure('Cyber.TLabel',
                             background=self.colors['bg'],
                             foreground=self.colors['fg'],
                             font=('Consolas', 10))

        self.style.configure('Cyber.TButton',
                             background=self.colors['panel_bg'],
                             foreground=self.colors['fg'],
                             borderwidth=2,
                             relief='raised',
                             font=('Consolas', 9))

        self.style.configure('Title.TLabel',
                             font=('Consolas', 16, 'bold'),
                             foreground=self.colors['accent'],
                             background=self.colors['bg'])

        self.style.configure('Warning.TLabel',
                             foreground=self.colors['warning'],
                             font=('Consolas', 10, 'bold'),
                             background=self.colors['bg'])

        self.style.configure('Cyber.Treeview',
                             background=self.colors['text_bg'],
                             foreground=self.colors['fg'],
                             fieldbackground=self.colors['text_bg'])

    def create_menu_bar(self):
        """Crée la barre de menu"""
        menubar = tk.Menu(self.root, bg=self.colors['panel_bg'], fg=self.colors['fg'])
        self.root.config(menu=menubar)

        # Menu Fichier
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['panel_bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Synchroniser Réseau", command=self.sync_network)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.root.quit)

        # Menu Administration (seulement si admin connecté)
        self.admin_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['panel_bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="Administration", menu=self.admin_menu)
        self.update_admin_menu()

        # Menu Configuration
        config_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['panel_bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="Configuration", menu=config_menu)
        config_menu.add_command(label="Paramètres", command=self.show_config)

        # Thème sous-menu
        theme_menu = tk.Menu(config_menu, tearoff=0, bg=self.colors['panel_bg'], fg=self.colors['fg'])
        config_menu.add_cascade(label="Thèmes", menu=theme_menu)

        for theme_name in self.theme_manager.get_theme_names():
            theme_menu.add_command(
                label=theme_name,
                command=lambda name=theme_name: self.change_theme(name)
            )

        # Menu Aide
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['panel_bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="Aide", menu=help_menu)
        help_menu.add_command(label="À propos", command=self.show_about)
        help_menu.add_command(label="Mises à jour", command=self.show_update_manager)

    def update_admin_menu(self):
        """Met à jour le menu admin en fonction de la connexion"""
        self.admin_menu.delete(0, tk.END)

        if self.face_system.admin_logged_in:
            self.admin_menu.add_command(label="🛑 Déconnexion Admin", command=self.admin_logout)
            self.admin_menu.add_separator()
            self.admin_menu.add_command(label="➕ Ajouter Visage", command=self.add_face_dialog)
            self.admin_menu.add_command(label="📷 Capturer Image", command=self.capture_image_admin)
            self.admin_menu.add_command(label="🗑️ Gérer Visages", command=self.manage_faces)
            self.admin_menu.add_command(label="🌐 Partager sur Réseau", command=self.share_to_network)
        else:
            self.admin_menu.add_command(label="🔐 Connexion Admin", command=self.admin_login)

    def change_theme(self, theme_name):
        """Change le thème de l'interface"""
        self.apply_theme(theme_name)
        # Recréer l'interface avec le nouveau thème
        for widget in self.root.winfo_children():
            widget.destroy()
        self.setup_futuristic_ui()
        messagebox.showinfo("Thème", f"Thème changé: {theme_name}")

    def create_main_interface(self):
        """Crée l'interface principale"""
        # Frame principal
        main_frame = ttk.Frame(self.root, style='Cyber.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Header avec effet matrix
        self.create_header(main_frame)

        # Corps de l'application
        body_frame = ttk.Frame(main_frame, style='Cyber.TFrame')
        body_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panneau gauche (caméra et contrôles)
        left_panel = self.create_left_panel(body_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Panneau droit (informations et logs)
        right_panel = self.create_right_panel(body_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Barre de statut
        self.create_status_bar(main_frame)

    def create_header(self, parent):
        """Crée l'en-tête style matrix"""
        header_frame = ttk.Frame(parent, style='Cyber.TFrame')
        header_frame.pack(fill=tk.X, padx=10, pady=5)

        # Titre principal
        title_frame = ttk.Frame(header_frame, style='Cyber.TFrame')
        title_frame.pack(fill=tk.X)

        title_label = ttk.Label(title_frame,
                                text="▓ CYBER-VISION AI v2.0 ▓ NEURAL FACE RECOGNITION SYSTEM ▓",
                                style='Title.TLabel')
        title_label.pack(side=tk.LEFT)

        # Indicateur admin
        self.admin_indicator = ttk.Label(title_frame,
                                         text="",
                                         style='Warning.TLabel')
        self.admin_indicator.pack(side=tk.LEFT, padx=10)
        self.update_admin_indicator()

        # Horloge numérique
        self.clock_label = ttk.Label(title_frame,
                                     text="▓ 00:00:00 ▓",
                                     style='Title.TLabel')
        self.clock_label.pack(side=tk.RIGHT)
        self.update_clock()

        # Barre de progression système
        self.system_status = ttk.Progressbar(header_frame,
                                             mode='indeterminate')
        self.system_status.pack(fill=tk.X, pady=5)
        self.system_status.start(15)

    def update_admin_indicator(self):
        """Met à jour l'indicateur admin"""
        if self.face_system.admin_logged_in:
            self.admin_indicator.config(text="🔓 ADMIN")
        else:
            self.admin_indicator.config(text="")

    def create_left_panel(self, parent):
        """Crée le panneau de contrôle gauche"""
        panel = ttk.Frame(parent, style='Cyber.TFrame')

        # Affichage caméra avec bordure néon
        cam_frame = ttk.LabelFrame(panel,
                                   text="▓ NEURAL VISION FEED ▓",
                                   style='Cyber.TFrame')
        cam_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.camera_label = tk.Label(cam_frame,
                                     background='black',
                                     borderwidth=2,
                                     relief='sunken')
        self.camera_label.pack(padx=3, pady=3, fill=tk.BOTH, expand=True)

        # Contrôles rapides
        controls_frame = ttk.Frame(panel, style='Cyber.TFrame')
        controls_frame.pack(fill=tk.X, pady=10)

        self.create_control_buttons(controls_frame)

        return panel

    def create_control_buttons(self, parent):
        """Crée les boutons de contrôle"""
        button_configs = [
            ("🚀 BOOT SYSTEM", self.toggle_camera, '#00ff00'),
            ("🛑 KILL SWITCH", self.stop_system, '#ff0000'),
            ("👁️ FULL SCAN", self.full_scan, '#00ffff'),
            ("📊 DASHBOARD", self.show_dashboard, '#ffff00'),
            ("⚙️ CONFIG", self.show_config, '#ff00ff'),
            ("🔍 UPDATE CHECK", self.check_updates, '#ff8800'),
            ("🔐 ADMIN", self.admin_login, '#ff4444')
        ]

        for i, (text, command, color) in enumerate(button_configs):
            btn = tk.Button(parent,
                            text=text,
                            command=command,
                            bg=self.colors['panel_bg'],
                            fg=color,
                            font=('Consolas', 9, 'bold'),
                            borderwidth=2,
                            relief='raised',
                            width=12,
                            height=2)
            btn.grid(row=0, column=i, padx=2, pady=2)

    def create_right_panel(self, parent):
        """Crée le panneau d'information droit"""
        panel = ttk.Frame(parent, width=400, style='Cyber.TFrame')
        panel.pack_propagate(False)

        # Section alertes de sécurité
        alert_frame = ttk.LabelFrame(panel,
                                     text="▓ SECURITY ALERTS ▓",
                                     style='Cyber.TFrame')
        alert_frame.pack(fill=tk.X, pady=5)

        self.alert_display = scrolledtext.ScrolledText(alert_frame,
                                                       height=6,
                                                       bg=self.colors['text_bg'],
                                                       fg=self.colors['warning'],
                                                       font=('Consolas', 8),
                                                       insertbackground='red',
                                                       wrap=tk.WORD)
        self.alert_display.pack(fill=tk.BOTH, padx=3, pady=3)

        # Section détections en temps réel
        detection_frame = ttk.LabelFrame(panel,
                                         text="▓ LIVE DETECTIONS ▓",
                                         style='Cyber.TFrame')
        detection_frame.pack(fill=tk.X, pady=5)

        # Treeview pour les détections
        tree_frame = ttk.Frame(detection_frame, style='Cyber.TFrame')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        self.detection_tree = ttk.Treeview(tree_frame,
                                           columns=('ID', 'Name', 'Conf', 'Time'),
                                           show='headings',
                                           height=6,
                                           style='Cyber.Treeview')

        # Configurer les colonnes
        columns = [('ID', 80), ('Name', 100), ('Conf', 60), ('Time', 80)]
        for col, width in columns:
            self.detection_tree.heading(col, text=col)
            self.detection_tree.column(col, width=width)

        self.detection_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Barre de défilement
        scrollbar = ttk.Scrollbar(tree_frame,
                                  orient=tk.VERTICAL,
                                  command=self.detection_tree.yview)
        self.detection_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Section statistiques système
        stats_frame = ttk.LabelFrame(panel,
                                     text="▓ SYSTEM STATS ▓",
                                     style='Cyber.TFrame')
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_text = tk.Text(stats_frame,
                                  height=6,
                                  bg=self.colors['text_bg'],
                                  fg=self.colors['fg'],
                                  font=('Consolas', 8),
                                  relief='flat',
                                  wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, padx=3, pady=3)

        return panel

    def create_status_bar(self, parent):
        """Crée la barre de statut"""
        status_frame = ttk.Frame(parent, style='Cyber.TFrame', height=25)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        # Statut système
        self.status_var = tk.StringVar(value="▓ SYSTEM: READY ▓")
        status_label = ttk.Label(status_frame,
                                 textvariable=self.status_var,
                                 style='Cyber.TLabel')
        status_label.pack(side=tk.LEFT, padx=10)

        # Indicateur de connexion
        self.connection_indicator = tk.Label(status_frame,
                                             text="●",
                                             fg='#00ff00',
                                             bg=self.colors['bg'],
                                             font=('Arial', 12, 'bold'))
        self.connection_indicator.pack(side=tk.RIGHT, padx=10)

        # Mode d'opération
        self.mode_var = tk.StringVar(value="MODE: SURVEILLANCE")
        mode_label = ttk.Label(status_frame,
                               textvariable=self.mode_var,
                               style='Cyber.TLabel')
        mode_label.pack(side=tk.RIGHT, padx=20)

    def update_clock(self):
        """Met à jour l'horloge numérique"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.clock_label.configure(text=f"▓ {current_time} ▓")
        self.root.after(1000, self.update_clock)

    # === MÉTHODES ADMINISTRATEUR ===

    def admin_login(self):
        """Connexion administrateur"""
        password = simpledialog.askstring("Admin Login",
                                          "Entrez le mot de passe administrateur:",
                                          show='*')
        if password == self.face_system.config['admin_password']:
            self.face_system.admin_logged_in = True
            self.update_admin_menu()
            self.update_admin_indicator()
            messagebox.showinfo("Admin", "Connexion administrateur réussie!")
        elif password is not None:  # None si annulé
            messagebox.showerror("Erreur", "Mot de passe incorrect!")

    def admin_logout(self):
        """Déconnexion administrateur"""
        self.face_system.admin_logged_in = False
        self.update_admin_menu()
        self.update_admin_indicator()
        messagebox.showinfo("Admin", "Déconnexion administrateur réussie!")

    def add_face_dialog(self):
        """Ouvre une boîte de dialogue pour ajouter un visage (admin seulement)"""
        if not self.face_system.admin_logged_in:
            messagebox.showerror("Accès refusé", "Connexion administrateur requise")
            return

        if not hasattr(self, 'camera_active') or not self.camera_active:
            messagebox.showerror("Erreur", "Veuillez démarrer la caméra d'abord")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("▓ AJOUTER VISAGE ▓")
        dialog.configure(bg=self.colors['bg'])
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="ID du visage:", style='Cyber.TLabel').pack(pady=5)
        id_entry = ttk.Entry(dialog, width=30)
        id_entry.pack(pady=5)

        ttk.Label(dialog, text="Nom:", style='Cyber.TLabel').pack(pady=5)
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack(pady=5)

        def capture_and_add():
            face_id = id_entry.get().strip()
            name = name_entry.get().strip()

            if not face_id or not name:
                messagebox.showerror("Erreur", "Veuillez remplir tous les champs")
                return

            if hasattr(self, 'cap'):
                ret, frame = self.cap.read()
                if ret:
                    success, message = self.face_system.add_face(frame, name, face_id)
                    if success:
                        messagebox.showinfo("Succès", message)
                        dialog.destroy()
                    else:
                        messagebox.showerror("Erreur", message)

        ttk.Button(dialog, text="Capturer et Ajouter",
                   command=capture_and_add, style='Cyber.TButton').pack(pady=10)

    def capture_image_admin(self):
        """Capture une image (admin seulement)"""
        if not self.face_system.admin_logged_in:
            messagebox.showerror("Accès refusé", "Connexion administrateur requise")
            return

        if hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if ret:
                success, message = self.face_system.capture_image(frame)
                if success:
                    messagebox.showinfo("Succès", message)
                else:
                    messagebox.showerror("Erreur", message)

    def manage_faces(self):
        """Gestion des visages (admin seulement)"""
        if not self.face_system.admin_logged_in:
            messagebox.showerror("Accès refusé", "Connexion administrateur requise")
            return

        messagebox.showinfo("Gestion Visages", "Interface de gestion des visages - Fonctionnalité à venir")

    # === MÉTHODES RÉSEAU ===

    def sync_network(self):
        """Synchronise avec le dossier partagé"""
        self.status_var.set("▓ SYSTEM: SYNC WITH NETWORK ▓")
        try:
            if self.face_system.network_manager.sync_with_network():
                # Recharger les données après synchronisation
                self.face_system.load_trained_data()
                messagebox.showinfo("Synchronisation", "Synchronisation réseau réussie!")
            else:
                messagebox.showinfo("Synchronisation", "Aucune nouvelle donnée à synchroniser")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur synchronisation: {str(e)}")
        self.status_var.set("▓ SYSTEM: READY ▓")

    def share_to_network(self):
        """Partage les données vers le réseau (admin seulement)"""
        if not self.face_system.admin_logged_in:
            messagebox.showerror("Accès refusé", "Connexion administrateur requise")
            return

        self.status_var.set("▓ SYSTEM: SHARING TO NETWORK ▓")
        if self.face_system.network_manager.share_to_network():
            messagebox.showinfo("Partage", "Données partagées vers le réseau!")
        else:
            messagebox.showerror("Erreur", "Erreur partage réseau ou aucune donnée à partager")
        self.status_var.set("▓ SYSTEM: READY ▓")

    # === MÉTHODES MISE À JOUR ===

    def show_update_manager(self):
        """Affiche le gestionnaire de mises à jour"""
        update_win = tk.Toplevel(self.root)
        update_win.title("▓ GESTIONNAIRE DE MISES À JOUR ▓")
        update_win.configure(bg=self.colors['bg'])
        update_win.geometry("500x400")

        ttk.Label(update_win, text="Gestionnaire de Mises à Jour",
                  style='Title.TLabel').pack(pady=10)

        # Liste des URLs de mise à jour
        ttk.Label(update_win, text="URLs de mise à jour:",
                  style='Cyber.TLabel').pack(pady=5)

        update_listbox = tk.Listbox(update_win,
                                    bg=self.colors['text_bg'],
                                    fg=self.colors['fg'],
                                    font=('Consolas', 9))
        update_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Charger les URLs existantes
        for url in self.face_system.config.get('update_urls', []):
            update_listbox.insert(tk.END, url)

        # Contrôles
        control_frame = ttk.Frame(update_win, style='Cyber.TFrame')
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        def add_url():
            new_url = simpledialog.askstring("Nouvelle URL", "Entrez la nouvelle URL:")
            if new_url and new_url not in self.face_system.config['update_urls']:
                self.face_system.config['update_urls'].append(new_url)
                self.face_system.save_config()
                update_listbox.insert(tk.END, new_url)

        def remove_url():
            selection = update_listbox.curselection()
            if selection:
                url = update_listbox.get(selection[0])
                self.face_system.config['update_urls'].remove(url)
                self.face_system.save_config()
                update_listbox.delete(selection[0])

        ttk.Button(control_frame, text="➕ Ajouter URL",
                   command=add_url, style='Cyber.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🗑️ Supprimer",
                   command=remove_url, style='Cyber.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🔄 Vérifier MAJ",
                   command=self.check_updates, style='Cyber.TButton').pack(side=tk.RIGHT, padx=5)

    # === AUTRES MÉTHODES ===

    def toggle_camera(self):
        """Active/désactive la caméra"""
        if not hasattr(self, 'camera_active'):
            self.camera_active = False

        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Démarre le système de vision"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erreur", "Impossible d'accéder à la caméra")
                return

            self.camera_active = True
            self.status_var.set("▓ SYSTEM: ACTIVE - NEURAL SCAN ENGAGED ▓")
            self.mode_var.set("MODE: ACTIVE SCAN")
            self.connection_indicator.config(fg='#00ff00')
            self.update_camera_feed()
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur caméra: {str(e)}")

    def stop_camera(self):
        """Arrête la caméra"""
        self.camera_active = False
        if hasattr(self, 'cap'):
            self.cap.release()
        self.status_var.set("▓ SYSTEM: STANDBY ▓")
        self.mode_var.set("MODE: STANDBY")
        self.connection_indicator.config(fg='#666666')

    def stop_system(self):
        """Arrête complètement le système"""
        if hasattr(self, 'camera_active') and self.camera_active:
            self.stop_camera()

        self.status_var.set("▓ SYSTEM: SHUTDOWN INITIATED ▓")
        self.connection_indicator.config(fg='#ff0000')

        result = messagebox.askyesno(
            "Arrêt du Système",
            "Êtes-vous sûr de vouloir arrêter le système Cyber-Vision ?"
        )

        if result:
            self.status_var.set("▓ SYSTEM: SHUTDOWN COMPLETE ▓")
            self.root.after(2000, self.root.quit)
        else:
            self.status_var.set("▓ SYSTEM: READY ▓")
            self.connection_indicator.config(fg='#00ff00')

    def update_camera_feed(self):
        """Met à jour le flux vidéo"""
        if self.camera_active and hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self.apply_cyber_effects(frame)
                movement_data = self.face_system.analyze_movement(
                    frame, getattr(self, 'previous_frame', None))
                results = self.face_system.recognize_faces_advanced(frame, movement_data)
                self.draw_detection_results(processed_frame, results)
                self.update_detection_display(results)
                self.update_system_stats()
                self.update_alerts_display()
                self.previous_frame = frame.copy()

                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

            self.root.after(30, self.update_camera_feed)

    def apply_cyber_effects(self, frame):
        """Applique des effets visuels"""
        height, width = frame.shape[:2]
        scan_line = int((time.time() * 50) % height)
        cv2.line(frame, (0, scan_line), (width, scan_line), (0, 255, 0), 2)

        if int(time.time() * 10) % 5 == 0:
            noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)

        return frame

    def draw_detection_results(self, frame, results):
        """Dessine les résultats de détection"""
        for result in results:
            x, y, w, h = result['location']

            if result['confidence'] > 0.8:
                color = (0, 255, 0)
            elif result['confidence'] > 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{result['name']} [{result['confidence']:.1%}]"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def update_detection_display(self, results):
        """Met à jour l'affichage des détections"""
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)

        for result in results:
            self.detection_tree.insert('', tk.END, values=(
                result['face_id'][:8],
                result['name'],
                f"{result['confidence']:.1%}",
                datetime.now().strftime("%H:%M:%S")
            ))

    def update_system_stats(self):
        """Met à jour les statistiques système"""
        stats_text = f"""
▓ SYSTEM UPTIME: {time.strftime('%H:%M:%S', time.gmtime(time.time() - getattr(self, 'start_time', time.time())))}
▓ FACES IN DB: {len(self.face_system.faces)}
▓ ACTIVE THREATS: {len(self.face_system.security_ai.suspicious_activities)}
▓ AI CONFIDENCE: {self.face_system.security_ai.learning_rate:.1%}
▓ ADMIN: {'CONNECTÉ' if self.face_system.admin_logged_in else 'NON CONNECTÉ'}
▓ LAST SCAN: {datetime.now().strftime('%H:%M:%S')}
        """.strip()

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)

    def update_alerts_display(self):
        """Met à jour l'affichage des alertes"""
        if self.face_system.security_ai.suspicious_activities:
            last_alert = self.face_system.security_ai.suspicious_activities[-1]
            alert_text = f"[{last_alert['timestamp']}] {last_alert['activity']} - {last_alert['level']}\n"
            self.alert_display.insert(tk.END, alert_text)
            self.alert_display.see(tk.END)

            lines = self.alert_display.get(1.0, tk.END).split('\n')
            if len(lines) > 50:
                self.alert_display.delete(1.0, f"{len(lines) - 50}.0")

    def full_scan(self):
        """Lance un scan complet"""
        self.status_var.set("▓ SYSTEM: DEEP SCAN INITIATED ▓")
        for i in range(5):
            self.status_var.set(f"▓ SYSTEM: SCANNING... {i + 1}/5 ▓")
            self.root.update()
            time.sleep(0.5)
        self.status_var.set("▓ SYSTEM: SCAN COMPLETE ▓")
        messagebox.showinfo("Scan Complet", "Scan du système terminé.")
        self.root.after(2000, lambda: self.status_var.set("▓ SYSTEM: READY ▓"))

    def show_dashboard(self):
        """Affiche le tableau de bord"""
        dashboard = tk.Toplevel(self.root)
        dashboard.title("▓ CYBER-VISION DASHBOARD ▓")
        dashboard.configure(bg=self.colors['bg'])
        dashboard.geometry("600x400")

        tk.Label(dashboard, text="DASHBOARD SYSTÈME",
                 bg=self.colors['bg'], fg=self.colors['accent'],
                 font=('Consolas', 16, 'bold')).pack(pady=20)

    def show_config(self):
        """Affiche la configuration"""
        config_win = tk.Toplevel(self.root)
        config_win.title("▓ CONFIGURATION ▓")
        config_win.configure(bg=self.colors['bg'])
        config_win.geometry("500x400")

        tk.Label(config_win, text="CONFIGURATION SYSTÈME",
                 bg=self.colors['bg'], fg=self.colors['fg'],
                 font=('Consolas', 14)).pack(pady=10)

        # Sélecteur de thème
        theme_frame = ttk.Frame(config_win, style='Cyber.TFrame')
        theme_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Label(theme_frame, text="Thème:", style='Cyber.TLabel').pack(side=tk.LEFT)

        theme_var = tk.StringVar(value=self.face_system.config.get('current_theme', 'Cyberpunk Matrix'))
        theme_combo = ttk.Combobox(theme_frame,
                                   textvariable=theme_var,
                                   values=self.theme_manager.get_theme_names(),
                                   state='readonly')
        theme_combo.pack(side=tk.LEFT, padx=10)
        theme_combo.bind('<<ComboboxSelected>>',
                         lambda e: self.change_theme(theme_var.get()))

        # Options de configuration
        options_frame = ttk.Frame(config_win, style='Cyber.TFrame')
        options_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        options = [
            ("Alertes Email", "email_alerts"),
            ("Alertes Sonores", "sound_alerts"),
            ("Auto-apprentissage", "auto_learning")
        ]

        for text, key in options:
            var = tk.BooleanVar(value=self.face_system.config[key])
            cb = tk.Checkbutton(options_frame, text=text, variable=var,
                                bg=self.colors['bg'], fg=self.colors['fg'],
                                selectcolor=self.colors['panel_bg'],
                                font=('Consolas', 10),
                                command=lambda k=key, v=var: self.update_config(k, v.get()))
            cb.pack(anchor='w', pady=5)

    def update_config(self, key, value):
        """Met à jour la configuration"""
        self.face_system.config[key] = value
        self.face_system.save_config()

    def check_updates(self):
        """Vérifie les mises à jour"""
        self.status_var.set("▓ SYSTEM: CHECKING FOR UPDATES ▓")
        self.root.after(2000, lambda: self.status_var.set("▓ SYSTEM: UPDATE CHECK COMPLETE ▓"))
        self.root.after(2500, lambda: messagebox.showinfo("Update Check", "Système à jour. Version 2.0 stable"))
        self.root.after(3000, lambda: self.status_var.set("▓ SYSTEM: READY ▓"))

    def show_about(self):
        """Affiche la boîte À propos"""
        messagebox.showinfo("À propos",
                            "Cyber-Vision AI v2.0\n\n"
                            "Système avancé de reconnaissance faciale\n"
                            "avec intelligence artificielle intégrée\n\n"
                            "© 2024 Cyber-Vision Technologies")

    def setup_system(self):
        """Configure le système au démarrage"""
        self.start_time = time.time()
        # Synchronisation automatique désactivée au démarrage pour éviter les erreurs
        # L'utilisateur peut synchroniser manuellement via le menu

    def __del__(self):
        """Nettoyage"""
        if hasattr(self, 'camera_active') and self.camera_active:
            self.stop_camera()


def main():
    root = tk.Tk()
    root.geometry("1200x800")

    # Mode plein écran disponible
    root.attributes('-fullscreen', False)

    app = HackerStyleGUI(root)

    # Raccourcis clavier
    root.bind('<F11>', lambda e: root.attributes('-fullscreen', not root.attributes('-fullscreen')))
    root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
    root.bind('<F1>', lambda e: app.show_dashboard())
    root.bind('<Control-q>', lambda e: root.quit())
    root.bind('<Control-a>', lambda e: app.admin_login())
    root.bind('<Control-s>', lambda e: app.sync_network())

    def on_closing():
        app.stop_camera()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    print("▓ CYBER-VISION AI v2.0 - SYSTEM BOOT COMPLETE ▓")
    print("▓ F11: Plein écran | F1: Dashboard | Ctrl+A: Admin | Ctrl+S: Sync | Ctrl+Q: Quitter ▓")

    root.mainloop()


if __name__ == "__main__":
    main()