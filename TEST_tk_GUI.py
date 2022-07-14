import tkinter as tk
from tkinter import ttk

root = tk.Tk()

tk.Label(root, text='Classic Label', fg='black').pack()
ttk.Label(root, text='Themed Label').pack()

root.mainloop()
