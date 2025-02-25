import tkinter as tk

class Overlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.7)
        self.root.overrideredirect(True)
        self.label = tk.Label(self.root, text="Frank initialized...", font=("Helvetica", 12), bg="black", fg="white")
        self.label.pack()

    def update_text(self, message):
        self.label.config(text=message)

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    overlay = Overlay()
    overlay.start()
