import tkinter as tk
from tkinter import scrolledtext, ttk
import queue
from src.config import CHAT_CONFIG

class ChatOverlay:
    def __init__(self, message_queue):
        self.root = tk.Tk()
        self.setup_window()
        self.message_queue = message_queue
        self.setup_ui()

    def setup_window(self):
        self.root.title(CHAT_CONFIG['window_title'])
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", "white")
        self.root.attributes("-fullscreen", True)
        self.root.overrideredirect(True)

    def setup_ui(self):
        style = ttk.Style()
        style.configure('Custom.TFrame', background='black')
        
        self.frame = ttk.Frame(self.root, padding="10", style='Custom.TFrame')
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.text_area = scrolledtext.ScrolledText(
            self.frame,
            wrap=tk.WORD,
            width=CHAT_CONFIG['width'],
            height=CHAT_CONFIG['height'],
            bg=CHAT_CONFIG['bg_color'],
            fg=CHAT_CONFIG['fg_color'],
            font=CHAT_CONFIG['font']
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)
        self.text_area.insert(tk.END, CHAT_CONFIG['welcome_message'])

    def update_chat(self):
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                self.text_area.insert(tk.END, f"{message}\n")
                self.text_area.yview(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.update_chat)

    def run(self):
        self.update_chat()
        self.root.mainloop()
