import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import pygame
import librosa
import numpy as np

model = load_model('Song_Predictor.keras')

def init_window():
    # Create the application window
    root = tk.Tk()
    root.title("Song Predictor")
    root.geometry("360x640")
    return root

def clear_window(root):
    for widget in root.winfo_children():
            widget.destroy()

def set_background():
    # Set background
    background_image = Image.open('frontend_assets/bg.jpg')
    background_image_with_alpha = background_image.copy()
    background_image_with_alpha.putalpha(50)  # Set transparency level (0: transparent, 255: opaque)
    background_image_tk = ImageTk.PhotoImage(background_image_with_alpha)
    background_label = tk.Label(root, image=background_image_tk)
    background_label.image = background_image_tk
    background_label.place(relwidth=1, relheight=1) 

def set_button(root, image_path, coordinates, command):
    button_image = Image.open(image_path)
    button_image_tk = ImageTk.PhotoImage(button_image)
    button = tk.Button(root, image=button_image_tk, bd=0, command=command)
    button.image = button_image_tk
    button.place(x=coordinates[0], y=coordinates[1])

def open_second_window(val):

    # onclick functions for this window
    def play_button_clicked():
        if pygame.mixer.music.get_busy():
           pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()
            time_str = time_label.cget("text")
            new_counter = int(time_str.split(":")[1])
            update_time(new_counter)

    def go_back_button_clicked():
        if pygame.mixer.music.get_busy():
           pygame.mixer.music.pause()
        for widget in root.winfo_children():
            widget.destroy()
        create_main_window()

    def map(val):
        mapping = {0:{'name':'Atlantis','artist':'Seafret'},
                1: {'name':'Babydoll','artist':'Ari Abdul'},
                2: {'name':'Dadada','artist':'Tanir and Tyomcha'},
                3: {'name':'Dancing','artist':'Aaron Smith'},
                4: {'name':'Dead Inside','artist':'АДЛИН'},
                5: {'name':'Georgian Disco', 'artist':'Nikos Band'},
                6: {'name':'In My Mind','artist':'Kenya Grace'},
                7: {'name':'La Espada', 'artist':'Eternal Raijin'},
                8: {'name':'Metamorphosis','artist':'INTERWORLD'},
                9: {'name':'Shadow Lady','artist':'Portwave'},
                10: {'name':'Somebody That I Used To Know','artist':'Gotye, Kimbra'},
                }
        return mapping[val]

    # time and seeker update
    def format_time(seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02}:{seconds:02}"
    def update_time(counter):
        if pygame.mixer.music.get_busy():
            counter += 1
            current_time = pygame.mixer.music.get_pos() / 1000
            time_label.config(text=format_time(current_time))
            time_label.after(1000, update_time, counter)
            update_line(counter)
    def update_line(counter):
        canvas.delete("green")
        x2_new = min((11.2*counter),345)
        canvas.create_line(15, 10, x2_new, 10, fill="green", width=2)

    # Define initial file path
    song = map(val)
    song_path = "frontend_assets/songs_and_covers/"+str(val)+".mp3"
    song_name, song_artist = song['name'], song['artist']
    set_background()
    
    # canvas for the image
    image_canvas = tk.Canvas(root, width=217, height=217)
    image_canvas.place(x=45+18.5+7.5, y=60.77)
    
    # canvas for the text
    text_canvas = tk.Canvas(root, width=360, height=100) 
    text_canvas.place(x=0, y=60.77 + 222 + 14)
    text_canvas.create_text(180, 40, anchor=tk.CENTER, justify=tk.CENTER, text=song_name, font=("Poppins", 25, "bold"), fill="green", width=360)
    text_canvas.create_text(180, 90, anchor=tk.CENTER, justify=tk.CENTER, text=song_artist, font=("Poppins", 10), fill="green", width=250)
    image_path = "frontend_assets/songs_and_covers/"+str(val)+".png" 
    image = Image.open(image_path)
    image = image.resize((217, 217))
    image_tk = ImageTk.PhotoImage(image)
    image_canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

    # Create canvas for seeker
    canvas = tk.Canvas(root, width=360, height=20)
    canvas.place(x=7.5, y=60.77+352+80-15)
    canvas.create_line(15, 10, 345, 10, fill="light grey", width=2) # base line

    set_button(root, "frontend_assets/SecondPage/play.png", (45+113, 60.77+352), play_button_clicked)
    set_button(root, "frontend_assets/SecondPage/go_back.png", (45+63, 60.77+352+127), go_back_button_clicked)

    # Load the music file to get its length
    pygame.mixer.init()
    pygame.mixer.music.load(song_path)
    pygame.mixer.Sound(song_path).get_length()

    # Label to display current time
    time_label = tk.Label(root, text="00:00")
    time_label.pack(pady=5)
    time_label.place(x=45+113+15,y= 60.77+352+80)

    pygame.mixer.music.play()

    # Start updating the time label and line
    update_time(0)
    root.mainloop()

def create_main_window():
    # onclick functions for this window
    def upload_button_clicked():
        file_path = filedialog.askopenfilename()
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def preprocess(song_path):
        test_inputs = np.zeros((1,15236))
        signal,sr = librosa.load(song_path)
        signal = np.resize(signal,600000)
        print(song_path)
        mfcc = librosa.feature.mfcc(y=signal,n_mfcc=13,sr=sr)
        test_inputs[0] = mfcc.flatten()
        return test_inputs
    
    def get_prediction_from_model(song_path):
        inputs = preprocess(song_path)
        val = model.predict(inputs)
        val = np.argmax(val)
        return val

    def continue_button_clicked():
        entry_text = entry.get()
        if not entry_text:
            tk.messagebox.showerror("Error", "Please select a file")
        else:
            val = get_prediction_from_model(entry_text)
            for widget in root.winfo_children():
                widget.destroy()
            open_second_window(val)

    # should i replace with vairable for cleaner look?
    entry = tk.Entry(root, width=42)
    entry.place(x=55+40-10-30, y=61.5+415.54-40)
    set_background()
    # logo
    logo_image_path = "frontend_assets/FirstPage/Logo.png"
    logo_image = Image.open(logo_image_path)
    logo_image_tk = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(root, image=logo_image_tk)
    logo_label.place(x=55, y=61.5)

    set_button(root, "frontend_assets/FirstPage/upload_button.png", (55+40-10, 61.5+415.54), upload_button_clicked)
    set_button(root, "frontend_assets/FirstPage/continue_button.png", (55+40+9-10, 61.5+415.54+62), continue_button_clicked)

    root.mainloop()

root = init_window()  # Call init_window once
create_main_window()