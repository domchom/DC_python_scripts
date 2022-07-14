import tkinter as tk
from tkinter import ttk

window = tk.Tk()

label =  tk.Label(text="paste the path to the folder with your movies:").pack()

entry = tk.Entry(fg="white", width=20).pack()


window.mainloop()


path = entry.get()

print(path)