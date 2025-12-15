import sys
import tkinter as tk
from tkinter.constants import CENTER
from tkinter import filedialog
from Utils.MidiParserForGUI import Parser
from midi2audio import FluidSynth
from Utils.Parser import Midi
import moviepy.editor as mpy
from Utils.video import midi_videoclip
import threading
import subprocess

# 指定 fluidsynth.exe 的完整路徑
FLUIDSYNTH_PATH = r"C:\fluidsynth-2.4.0-win10-x64\bin\fluidsynth.exe"

class CustomFluidSynth(FluidSynth):
    def midi_to_audio(self, midi_file, audio_file):
        # 使用完整路徑調用 fluidsynth
        subprocess.call([FLUIDSYNTH_PATH, '-ni', self.sound_font, midi_file, '-F', audio_file, '-r', str(self.sample_rate)])

class RedirectText(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.update_idletasks()  # Ensure the widget updates immediately
        self.text_widget.see(tk.END)  # Scroll to the end

    def flush(self):
        pass

def loadmidiFile():
    file_path = filedialog.askopenfilename(filetypes = (("midi files","*.mid"),("all files","*.*")))
    if file_path:
        loadFile_en.delete(0, tk.END)  # 清空輸入框
        loadFile_en.insert(0, file_path) 
    loadFile_en.xview_moveto(1)

def loadcsvFiles():
    file_paths = filedialog.askopenfilenames(filetypes=(("csv files", "*.csv"), ("all files", "*.*")))

    if file_paths:
        # 讀取當前文本框的內容，並在後面追加選中的文件路徑
        current_text = loadFile_en2.get("1.0", tk.END).strip()
        new_text = '\n'.join(file_paths)
        # 重新設置文本框內容，確保不覆蓋原來的路徑
        if current_text:
            loadFile_en2.delete("1.0", tk.END)
            loadFile_en2.insert(tk.END, current_text + '\n' + new_text)
        else:
            loadFile_en2.insert(tk.END, new_text + '\n')

def loadSoundFontFile():
    file_path = filedialog.askopenfilename(filetypes=(("sound font files", "*.sf2"), ("all files", "*.*")))
    if file_path:
        loadSoundFont_en.delete(0, tk.END)  # 清空輸入框
        loadSoundFont_en.insert(0, file_path)

def output():
    midi_file_path = loadFile_en.get()
    csv_file_paths = loadFile_en2.get("1.0", tk.END).strip().split('\n')
    sound_font_path = loadSoundFont_en.get()  # 获取音色文件路径

    midi_file_name = midi_file_path.split('/')[-1].split('.')[0]
    parser = Parser(midi_file_name)
    parser.judgeInsrtumentType()
    parser.getEvents()

    # 使用 CustomFluidSynth 進行音頻轉換
    fs = CustomFluidSynth(sound_font_path)
    fs.midi_to_audio(midi_file_path, f'{midi_file_name}.wav')

    midi = Midi(midi_file_name, csv_file_paths)
    clip = midi_videoclip(midi)
    audio_clip = mpy.AudioFileClip(f'{midi_file_name}.wav').set_duration(clip.duration)
    clip_with_audio = clip.set_audio(audio_clip)
    clip_with_audio.write_videofile(f'{midi_file_name}.mp4', fps=30, audio=True)

    subprocess.run(["ffplay", "-x", "1600", "-y", "900", f"{midi_file_name}.mp4"])

def start_output_thread():
    output_thread = threading.Thread(target=output)
    output_thread.start()

win = tk.Tk()
win.title('Piano MIDI visualization and Fingering comparison video generator')
win.geometry('680x300')
win.resizable(False, False)

lb = tk.Label(text="Choose MIDI file:", bg="grey", fg="white", height=1)
lb.place(x=23, y=0)
lb2 = tk.Label(text="Choose 4 CSV files (L/R's true/predicted):", bg="grey", fg="white", height=1)
lb2.place(x=23, y=25)
lb3 = tk.Label(text="Choose sf2 file:", bg="grey", fg="white", height=1)
lb3.place(x=23, y=100)

loadFile_en = tk.Entry(width=70)
loadFile_en.place(x=130, y=0)
loadFile_en2 = tk.Text(width=50, height=4)
loadFile_en2.place(x=265, y=25)
loadSoundFont_en = tk.Entry(width=70)
loadSoundFont_en.place(x=122, y=100)

loadFile_btn = tk.Button(text="...", height=1, command=loadmidiFile)
loadFile_btn.place(x=635, y=0)
loadFile_btn2 = tk.Button(text="...", height=1, command=loadcsvFiles)
loadFile_btn2.place(x=635, y=25)
loadSoundFont_btn = tk.Button(text="...", height=1, command=loadSoundFontFile)
loadSoundFont_btn.place(x=635, y=100)

output_btn = tk.Button(text="Go", height=1, command=start_output_thread)
output_btn.place(anchor=CENTER, x=650, y=170)

terminal_output = tk.Text(win, height=10, width=85)
terminal_output.place(x=20, y=150)
sys.stdout = RedirectText(terminal_output)

win.mainloop()