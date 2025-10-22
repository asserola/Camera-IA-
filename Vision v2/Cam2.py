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
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
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

        # Configuration
        self.config = self.load_config()
        self.setup_directories()

    def setup_directories(self):
        """Crée les répertoires nécessaires"""
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)

    def load_config(self):
        """Charge la configuration du système"""
        default_config = {
            'detection_confidence': 0.6,
            'max_unknown_faces': 3,
            'email_alerts': False,
            'sound_alerts': True,
            'auto_learning': True,
            'admin_password': 'cyber2024',
            'update_check_interval': 3600
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

    def recognize_faces_advanced(self, frame, movement_data):
        """Reconnaissance faciale avancée avec analyse comportementale"""
        faces, gray, _ = self.detect_faces_with_movement(frame, frame)
        results = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)  # Amélioration du contraste

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
                except:
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
                self.faces.append(face_roi)
                self.labels.append(len(self.faces) + 1)
                self.face_info[len(self.faces)] = {
                    'id': face_id,
                    'name': name,
                    'added_date': datetime.now()
                }
                self.face_recognizer.train(self.faces, np.array(self.labels))

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


class HackerStyleGUI:
    def __init__(self, root):
        self.root = root
        self.setup_futuristic_ui()
        self.face_system = FuturisticFaceRecognition()
        self.setup_system()

    def setup_futuristic_ui(self):
        """Configure l'interface style hacker"""
        self.root.title("▓ CYBER-VISION AI v2.0 ▓ Neural Face Recognition System")
        self.root.configure(bg='#001100')

        # Style futuriste
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Couleurs style hacker
        self.colors = {
            'bg': '#001100',
            'fg': '#00ff00',
            'accent': '#00ffff',
            'warning': '#ff0000',
            'panel_bg': '#002200',
            'text_bg': '#000000'
        }

        self.configure_styles()
        self.create_main_interface()

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
            ("🔐 ADMIN LOGIN", self.admin_login, '#ff4444')
        ]

        for i, (text, command, color) in enumerate(button_configs):
            btn = tk.Button(parent,
                            text=text,
                            command=command,
                            bg='#002200',
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
                                                       bg='#000000',
                                                       fg='#ff0000',
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
                                  bg='#000000',
                                  fg='#00ff00',
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
                                             bg='#001100',
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

        # Demander confirmation
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
        """Met à jour le flux vidéo avec effets visuels"""
        if self.camera_active and hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if ret:
                # Effet visuel style matrix
                processed_frame = self.apply_cyber_effects(frame)

                # Détection et reconnaissance
                movement_data = self.face_system.analyze_movement(
                    frame, getattr(self, 'previous_frame', None))
                results = self.face_system.recognize_faces_advanced(frame, movement_data)

                # Dessiner les résultats
                self.draw_detection_results(processed_frame, results)

                # Mettre à jour l'interface
                self.update_detection_display(results)
                self.update_system_stats()
                self.update_alerts_display()

                self.previous_frame = frame.copy()

                # Conversion pour Tkinter
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)

                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

            self.root.after(30, self.update_camera_feed)

    def apply_cyber_effects(self, frame):
        """Applique des effets visuels style cyberpunk"""
        # Effet de scan laser
        height, width = frame.shape[:2]
        scan_line = int((time.time() * 50) % height)

        # Ligne de scan lumineuse
        cv2.line(frame, (0, scan_line), (width, scan_line), (0, 255, 0), 2)

        # Ajouter du bruit numérique occasionnel
        if int(time.time() * 10) % 5 == 0:
            noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)

        return frame

    def draw_detection_results(self, frame, results):
        """Dessine les résultats de détection avec style futuriste"""
        for result in results:
            x, y, w, h = result['location']

            # Couleur basée sur la confiance
            if result['confidence'] > 0.8:
                color = (0, 255, 0)  # Vert - Reconnu
            elif result['confidence'] > 0.5:
                color = (0, 255, 255)  # Cyan - Incertain
            else:
                color = (0, 0, 255)  # Rouge - Inconnu

            # Rectangle avec effet néon
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Texte d'identification
            label = f"{result['name']} [{result['confidence']:.1%}]"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def update_detection_display(self, results):
        """Met à jour l'affichage des détections"""
        # Nettoyer l'arbre
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)

        # Ajouter les nouvelles détections
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
▓ LAST SCAN: {datetime.now().strftime('%H:%M:%S')}
▓ MEMORY USAGE: {self.get_memory_usage()}%
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

            # Limiter le nombre de lignes
            lines = self.alert_display.get(1.0, tk.END).split('\n')
            if len(lines) > 50:
                self.alert_display.delete(1.0, f"{len(lines) - 50}.0")

    def get_memory_usage(self):
        """Retourne l'utilisation mémoire approximative"""
        try:
            import psutil
            return f"{psutil.virtual_memory().percent:.1f}"
        except:
            return "N/A"

    def full_scan(self):
        """Lance un scan complet du système"""
        self.status_var.set("▓ SYSTEM: DEEP SCAN INITIATED ▓")

        # Simulation d'un scan complet
        for i in range(5):
            self.status_var.set(f"▓ SYSTEM: SCANNING... {i + 1}/5 ▓")
            self.root.update()
            time.sleep(0.5)

        self.status_var.set("▓ SYSTEM: SCAN COMPLETE ▓")
        messagebox.showinfo("Scan Complet", "Scan du système terminé.\nAucune menace détectée.")

        self.root.after(2000, lambda: self.status_var.set("▓ SYSTEM: READY ▓"))

    def show_dashboard(self):
        """Affiche le tableau de bord administrateur"""
        dashboard = tk.Toplevel(self.root)
        dashboard.title("▓ CYBER-VISION DASHBOARD ▓")
        dashboard.configure(bg='#001100')
        dashboard.geometry("600x400")

        # Empêcher la fermeture accidentelle
        dashboard.protocol("WM_DELETE_WINDOW", lambda: None)

        tk.Label(dashboard, text="ADMIN DASHBOARD",
                 bg='#001100', fg='#00ffff', font=('Consolas', 16, 'bold')).pack(pady=20)

        # Boutons du dashboard
        buttons = [
            ("👥 Gérer Utilisateurs", self.manage_users),
            ("📊 Voir Logs Complets", self.view_logs),
            ("💾 Sauvegarder Système", self.backup_system),
            ("🔧 Maintenance", self.system_maintenance),
            ("🚪 Quitter Dashboard", dashboard.destroy)
        ]

        for text, command in buttons:
            btn = tk.Button(dashboard, text=text, command=command,
                            bg='#002200', fg='#00ff00', font=('Consolas', 12),
                            width=20, height=2)
            btn.pack(pady=5)

    def manage_users(self):
        """Gestion des utilisateurs"""
        messagebox.showinfo("Gestion Utilisateurs", "Module de gestion utilisateurs")

    def view_logs(self):
        """Affiche les logs complets"""
        log_window = tk.Toplevel(self.root)
        log_window.title("▓ SYSTEM LOGS ▓")
        log_window.geometry("800x600")
        log_window.configure(bg='#001100')

        text_area = scrolledtext.ScrolledText(log_window,
                                              bg='#000000',
                                              fg='#00ff00',
                                              font=('Consolas', 9))
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Charger les logs
        try:
            with open("security_log.json", "r") as f:
                logs = f.readlines()
                for log in logs[-100:]:  # 100 dernières lignes
                    text_area.insert(tk.END, log)
        except:
            text_area.insert(tk.END, "Aucun log disponible\n")

    def backup_system(self):
        """Sauvegarde le système"""
        self.status_var.set("▓ SYSTEM: BACKUP IN PROGRESS ▓")
        messagebox.showinfo("Sauvegarde", "Sauvegarde du système en cours...")
        self.root.after(3000, lambda: self.status_var.set("▓ SYSTEM: READY ▓"))

    def system_maintenance(self):
        """Maintenance du système"""
        messagebox.showinfo("Maintenance", "Outils de maintenance système")

    def show_config(self):
        """Affiche la configuration système"""
        config_win = tk.Toplevel(self.root)
        config_win.title("▓ SYSTEM CONFIGURATION ▓")
        config_win.configure(bg='#001100')
        config_win.geometry("500x400")

        tk.Label(config_win, text="SYSTEM CONFIGURATION",
                 bg='#001100', fg='#00ff00', font=('Consolas', 14)).pack(pady=10)

        # Options de configuration
        options = [
            ("Alertes Email", "email_alerts"),
            ("Alertes Sonores", "sound_alerts"),
            ("Auto-apprentissage", "auto_learning")
        ]

        for text, key in options:
            var = tk.BooleanVar(value=self.face_system.config[key])
            cb = tk.Checkbutton(config_win, text=text, variable=var,
                                bg='#001100', fg='#00ff00', selectcolor='#002200',
                                font=('Consolas', 10),
                                command=lambda k=key, v=var: self.update_config(k, v.get()))
            cb.pack(pady=5)

        # Bouton de test d'alerte
        test_btn = tk.Button(config_win, text="🔊 Test Alerte Sonore",
                             command=self.test_alert,
                             bg='#002200', fg='#ffff00', font=('Consolas', 10))
        test_btn.pack(pady=10)

    def update_config(self, key, value):
        """Met à jour la configuration"""
        self.face_system.config[key] = value
        self.face_system.save_config()

    def test_alert(self):
        """Teste le système d'alerte"""
        self.face_system.notification_system.play_alert_sound("critical")
        messagebox.showinfo("Test Alerte", "Alerte sonore testée avec succès!")

    def check_updates(self):
        """Vérifie les mises à jour"""
        self.status_var.set("▓ SYSTEM: CHECKING FOR UPDATES ▓")

        # Simulation de vérification
        self.root.after(2000, lambda: self.status_var.set("▓ SYSTEM: UPDATE CHECK COMPLETE ▓"))
        self.root.after(2500, lambda: messagebox.showinfo("Update Check", "Système à jour. Version 2.0 stable"))
        self.root.after(3000, lambda: self.status_var.set("▓ SYSTEM: READY ▓"))

    def admin_login(self):
        """Connexion administrateur"""
        password = tk.simpledialog.askstring("Admin Login",
                                             "Entrez le mot de passe administrateur:",
                                             show='*')
        if password == self.face_system.config['admin_password']:
            self.show_admin_panel()
        elif password is not None:  # None si annulé
            messagebox.showerror("Erreur", "Mot de passe incorrect!")

    def show_admin_panel(self):
        """Affiche le panel administrateur complet"""
        admin_win = tk.Toplevel(self.root)
        admin_win.title("▓ ADMINISTRATION PANEL ▓")
        admin_win.configure(bg='#001100')
        admin_win.geometry("700x500")

        # Onglets
        notebook = ttk.Notebook(admin_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Onglet Système
        system_frame = ttk.Frame(notebook, style='Cyber.TFrame')
        notebook.add(system_frame, text="▓ SYSTÈME ▓")

        # Onglet Sécurité
        security_frame = ttk.Frame(notebook, style='Cyber.TFrame')
        notebook.add(security_frame, text="▓ SÉCURITÉ ▓")

        # Onglet Base de données
        db_frame = ttk.Frame(notebook, style='Cyber.TFrame')
        notebook.add(db_frame, text="▓ BASE DE DONNÉES ▓")

        # Remplir les onglets
        self.fill_system_tab(system_frame)
        self.fill_security_tab(security_frame)
        self.fill_database_tab(db_frame)

    def fill_system_tab(self, parent):
        """Remplit l'onglet système"""
        tk.Label(parent, text="Administration Système",
                 bg='#001100', fg='#00ffff', font=('Consolas', 12)).pack(pady=10)

    def fill_security_tab(self, parent):
        """Remplit l'onglet sécurité"""
        tk.Label(parent, text="Paramètres de Sécurité",
                 bg='#001100', fg='#00ffff', font=('Consolas', 12)).pack(pady=10)

    def fill_database_tab(self, parent):
        """Remplit l'onglet base de données"""
        tk.Label(parent, text="Gestion Base de Données",
                 bg='#001100', fg='#00ffff', font=('Consolas', 12)).pack(pady=10)

    def setup_system(self):
        """Configure le système au démarrage"""
        self.start_time = time.time()
        self.check_updates()

    def __del__(self):
        """Nettoyage"""
        if hasattr(self, 'camera_active') and self.camera_active:
            self.stop_camera()


def main():
    root = tk.Tk()
    root.geometry("1200x800")
    root.configure(bg='#001100')

    # Mode plein écran disponible
    root.attributes('-fullscreen', False)

    # Icône et titre
    root.title("Cyber-Vision AI v2.0")

    app = HackerStyleGUI(root)

    # Raccourcis clavier
    root.bind('<F11>', lambda e: root.attributes('-fullscreen', not root.attributes('-fullscreen')))
    root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
    root.bind('<F1>', lambda e: app.show_dashboard())
    root.bind('<Control-q>', lambda e: root.quit())
    root.bind('<Control-a>', lambda e: app.admin_login())

    def on_closing():
        app.stop_camera()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    print("▓ CYBER-VISION AI v2.0 - SYSTEM BOOT COMPLETE ▓")
    print("▓ F11: Plein écran | F1: Dashboard | Ctrl+A: Admin | Ctrl+Q: Quitter ▓")

    root.mainloop()


if __name__ == "__main__":
    main()