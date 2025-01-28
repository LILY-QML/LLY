import threading
import time
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os


class ThreadMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Singleton-Pattern, um sicherzustellen, dass nur eine Instanz des Monitors existiert.
        """
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ThreadMonitor, cls).__new__(cls)
                    cls._instance.thread_data = []
                    cls._instance.lock = threading.Lock()
        return cls._instance

    def log_thread_start(self, thread_id, process_name):
        """
        Protokolliert den Start eines Threads.

        :param thread_id: Die eindeutige ID des Threads.
        :param process_name: Der Name des Prozesses (z. B. Optimierung eines Qubits).
        """
        start_time = time.time()
        with self.lock:
            self.thread_data.append({
                "thread_id": thread_id,
                "process_name": process_name,
                "start_time": start_time,
                "end_time": None,
                "duration": None
            })

    def log_thread_end(self, thread_id):
        """
        Protokolliert das Ende eines Threads.

        :param thread_id: Die eindeutige ID des Threads.
        """
        end_time = time.time()
        with self.lock:
            for entry in self.thread_data:
                if entry["thread_id"] == thread_id and entry["end_time"] is None:
                    entry["end_time"] = end_time
                    entry["duration"] = end_time - entry["start_time"]
                    break

    def generate_pdf_report(self, folder="log", filename="Thread.pdf"):
        """
        Erstellt einen PDF-Bericht basierend auf den gesammelten Thread-Daten.

        :param folder: Der Ordner, in dem die PDF gespeichert wird.
        :param filename: Der Dateiname der generierten PDF-Datei.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        filepath = os.path.join(folder, filename)
        c = canvas.Canvas(filepath, pagesize=A4)
        width, height = A4

        # Titel
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Thread Activity Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Tabelle mit Thread-Daten
        c.drawString(50, height - 100, "Thread Overview:")
        y = height - 120
        c.setFont("Helvetica", 10)

        headers = ["Thread ID", "Process Name", "Start Time", "End Time", "Duration (s)"]
        c.drawString(50, y, " | ".join(headers))
        y -= 20

        for entry in self.thread_data:
            start_time = time.strftime('%H:%M:%S', time.localtime(entry["start_time"]))
            end_time = time.strftime('%H:%M:%S', time.localtime(entry["end_time"])) if entry["end_time"] else "N/A"
            duration = f"{entry['duration']:.2f}" if entry["duration"] else "N/A"

            line = f"{entry['thread_id']} | {entry['process_name']} | {start_time} | {end_time} | {duration}"
            c.drawString(50, y, line)
            y -= 15

            if y < 50:  # Neue Seite bei Platzmangel
                c.showPage()
                y = height - 50

        c.save()

    def __str__(self):
        """
        Gibt eine Übersicht der gesammelten Thread-Daten als String zurück.
        """
        overview = ["Thread Overview:"]
        for entry in self.thread_data:
            overview.append(f"Thread ID: {entry['thread_id']}, Process: {entry['process_name']}, "
                            f"Start: {time.strftime('%H:%M:%S', time.localtime(entry['start_time']))}, "
                            f"End: {time.strftime('%H:%M:%S', time.localtime(entry['end_time'])) if entry['end_time'] else 'N/A'}, "
                            f"Duration: {entry['duration']:.2f}s" if entry['duration'] else "N/A")
        return "\n".join(overview)
