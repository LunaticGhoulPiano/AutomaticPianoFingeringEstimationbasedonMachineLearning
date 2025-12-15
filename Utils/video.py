import gizeh
import moviepy.editor as mpy
import numpy as np
from moviepy.editor import *
from heapq import heappush, heappop
from more_itertools import peekable, first
from Utils.Parser import Note
from itertools import count
from PIL import Image
import tkinter as tk
from tkinter import filedialog

IS_IVORY_KEYS = [x not in [1, 3, 6, 8, 10] for x in range(12)]

is_ebony = lambda pitch: not IS_IVORY_KEYS[pitch % 12]
is_ivory = lambda pitch: IS_IVORY_KEYS[pitch % 12]

NEAR_EBONY_KEYS = [[] for _ in range(12)]
for ebony in [1, 3, 6, 8, 10]:
    NEAR_EBONY_KEYS[ebony + 1].append(-1)
    NEAR_EBONY_KEYS[ebony - 1].append(1)

OFFSET = [0.0] * 110

ivory = filter(is_ivory, range(21, 109))
ivory_offsets = count(start=0.5)
ebony = filter(is_ebony, range(21, 109))
ebony_offsets = filter(lambda x: x % 7 not in [2, 5], count(start=1))

for pitch, off in zip(ivory, ivory_offsets):
    OFFSET[pitch] = off / 52

for pitch, off in zip(ebony, ebony_offsets):
    OFFSET[pitch] = off / 52

PALETTE = {
  'ivory': [
    (0.79, 0.80, 0.79),  #CBCFCC, for white keys
  ],
  'ebony': [
    (0.1, 0.1, 0.1),  #3A3E42, for black keys
  ],
  'Note': [
    (0.9804, 0.2824, 0.5725, 0.8),  #5F819D, for each finger
    (0.9804, 0.2863, 0.2824, 0.8),  #3A3E42
    (0.9765, 0.8745, 0.2824, 0.8),  #DE935F
    (0.5176, 0.9804, 0.2824, 0.8),  #5E8D87
    (0.1059, 0.5098, 0.9765, 0.8),  #85678F
    (0.7608, 0.2902, 0.9804, 0.8), #5CB5E6
    (0.9765, 0.5412, 0.2824, 0.8), #F88348
    (0.3647, 0.7882, 0.9804, 0.8), #48F9A1
    (0.2824, 0.9804, 0.6275, 0.8), #F94892
    (0.9765, 0.8824, 0.2824, 0.8), #F9E249
  ],
  'Bar': [
    (0.9804, 0.2824, 0.5725),  #5F819D, for each finger
    (0.9804, 0.2863, 0.2824),  #3A3E42
    (0.9765, 0.8745, 0.2824),  #DE935F
    (0.5176, 0.9804, 0.2824),  #5E8D87
    (0.1059, 0.5098, 0.9765),  #85678F
    (0.7608, 0.2902, 0.9804), #5CB5E6
    (0.9765, 0.5412, 0.2824), #F88348
    (0.3647, 0.7882, 0.9804), #48F9A1
    (0.2824, 0.9804, 0.6275), #F94892
    (0.9765, 0.8824, 0.2824), #F9E249
  ]
}

POSITION = [
    (304, 199),  #-1
    (354, 199),  #1
    (414, 124),  #2
    (479, 91),  #3
    (539, 114),  #4
    (587, 155), #5
    (67, 155),  #-5
    (117, 114),  #-4
    (178, 91),  #-3
    (242, 124),  #-2
]

class ForeseePart: # draw note pics
    def __init__(self, midi, size):
        self.midi = midi
        self.size = size
        self.notes = list()
        all_notes = [Note(i[0], i[1], i[2]) for i in midi.timeline.items()]
        all_notes = sorted(all_notes, key=lambda n: n.begin)
        self.waits = peekable(all_notes)
        self.foresee = 2  # sec

    def make_frame(self, time):
        now = time
        future = time + self.foresee
        NONE = Note(float('inf'), float('inf'), 0)

        while first(self.notes, NONE).end < now:
            heappop(self.notes)
        while self.waits.peek(NONE).begin <= future:
            note = next(self.waits)
            heappush(self.notes, note)

        surface = gizeh.Surface(*self.size)
        for note in self.notes:
            rect = self.spawn_rectangle(note, now, future)
            rect.draw(surface)
        return surface.get_npimage()


    def spawn_rectangle(self, note, now, future):
        w, h = self.size
        begin, end = max(note.begin, now), min(note.end, future)
        pitch = self.midi.notes[note.index]['note']
        track = self.midi.notes[note.index]['track']
        track12 = self.midi.notes[note.index]['track12']
        switch = self.midi.notes[note.index]['switch']

        material = 'Bar'
        color = PALETTE[material][track % 10]

        lx = w / 52 if is_ivory(pitch) else w / 52 * 0.7
        ly = h * (end - begin) / (future - now) - 5
        xy = (w * OFFSET[pitch],
              h * (future - end / 2 - begin / 2) / (future - now))
        fill = color
        stroke = PALETTE['ebony'][-1]
        base_rect = gizeh.rectangle(lx=lx, ly=ly, xy=xy, fill=fill, stroke=stroke, stroke_width=1)
    
        if (track != track12):
            color = PALETTE[material][track12 % 10]
            lx = w / 52 if is_ivory(pitch) else w / 52 * 0.7
            ly = h * (end - switch) / (future - now) - 5
            xy = (w * OFFSET[pitch],
                h * (future - end / 2 - switch / 2) / (future - now))
            top_rect = gizeh.rectangle(lx=lx, ly=ly, xy=xy, fill=color, stroke=stroke, stroke_width=1)
            rect = gizeh.Group([base_rect, top_rect])
            return rect
        else :
            return base_rect

class PianoPart: # draw keyboard
    def __init__(self, midi, size):
        self.midi = midi
        self.size = size
        self.notes = list()
        all_notes = [Note(i[0], i[1], i[2]) for i in midi.timeline.items()]
        all_notes = sorted(all_notes, key=lambda n: n.begin)
        self.waits = peekable(all_notes)
        self.idle_piano = self.init_idle_piano()

    def make_frame(self, time):
        now = time
        NONE = Note(float('inf'), float('inf'), 0)

        while first(self.notes, NONE).end < now:
            heappop(self.notes)
        while self.waits.peek(NONE).begin <= now:
            note = next(self.waits)
            heappush(self.notes, note)

        redraw_ivory = {}
        redraw_ebony = {}
        for note in self.notes:
            pitch = self.midi.notes[note.index]['note']
            if is_ivory(pitch):
                redraw_ivory[pitch] = note
                for neighbor in NEAR_EBONY_KEYS[pitch % 12]:
                    if pitch + neighbor not in redraw_ebony:
                        redraw_ebony[pitch + neighbor] = None
            else:
                redraw_ebony[pitch] = note

        surface = gizeh.Surface(*self.size)
        arr = np.frombuffer(surface._cairo_surface.get_data(), np.uint8)
        arr += self.idle_piano
        surface._cairo_surface.mark_dirty()

        for pitch, note in redraw_ivory.items():
            rect = self.spawn_ivory_key(pitch, note, now)
            rect.draw(surface)
        for pitch, note in redraw_ebony.items():
            rect = self.spawn_ebony_key(pitch, note, now)
            rect.draw(surface)

        return surface.get_npimage()

    def init_idle_piano(self):
        surface = gizeh.Surface(*self.size)
        for pitch in filter(is_ivory, range(21, 109)):
            rect = self.spawn_ivory_key(pitch, None, None)
            rect.draw(surface)
        for pitch in filter(is_ebony, range(21, 109)):
            rect = self.spawn_ebony_key(pitch, None, None)
            rect.draw(surface)

        w, h = self.size
        image = surface.get_npimage()
        image = image[:, :, [2, 1, 0]]
        image = np.dstack([image, 255 * np.ones((h, w), dtype=np.uint8)])
        image = image.flatten()
        return image

    def spawn_ivory_key(self, pitch, note=None, now=None):
        w, h = self.size
        color = PALETTE['ivory'][-1]
        stroke_width = 1
        if note:
            pitch = self.midi.notes[note.index]['note']
            track = self.midi.notes[note.index]['track']
            color = PALETTE['Note'][track % 10]
            switch = self.midi.notes[note.index]['switch']
            track12 = self.midi.notes[note.index]['track12']
            stroke_width = 1.5
            if (track != track12):
                if now > switch:
                    color = PALETTE['Note'][track12 % 10]

        lx = w / 52
        ly = h
        xy = (w * OFFSET[pitch], h / 2)
        fill = color
        stroke = PALETTE['ebony'][-1]

        ivory_key_rect = gizeh.rectangle(lx=lx, ly=ly, xy=xy, fill=fill, stroke=stroke, stroke_width=stroke_width)
        
        return ivory_key_rect

    def spawn_ebony_key(self, pitch, note=None, now = None):
        w, h = self.size
        color = PALETTE['ebony'][-1]
        stroke_width = 1
        if note:
            pitch = self.midi.notes[note.index]['note']
            track = self.midi.notes[note.index]['track']
            switch = self.midi.notes[note.index]['switch']
            track12 = self.midi.notes[note.index]['track12']
            color = PALETTE['Note'][track % 10]
            stroke_width = 1.5
            if (track != track12):
                if now > switch:
                    color = PALETTE['Note'][track12 % 10]

        lx = w / 52 * 0.7
        ly = h * 2 / 3
        xy = (w * OFFSET[pitch], h / 3)
        fill = color
        stroke = PALETTE['ebony'][-1]

        ebony_key_rect = gizeh.rectangle(lx=lx, ly=ly, xy=xy, fill=fill, stroke=stroke, stroke_width=stroke_width)


        return ebony_key_rect

class HandPart: # draw hands
    def __init__(self, midi, image):
        self.midi = midi
        self.size = image.size
        self.notes = list()
        all_notes = [Note(i[0], i[1], i[2]) for i in midi.timeline.items()]
        all_notes = sorted(all_notes, key=lambda n: n.begin)
        self.waits = peekable(all_notes)
        self.image = image
        self.checknum = 0
        self.checked_notes = set()
        self.correct = 0
        self.total = midi.total
        
    def make_frame(self, time):
        now = time
        NONE = Note(float('inf'), float('inf'), 0)

        while first(self.notes, NONE).end < now:
            heappop(self.notes)
        while self.waits.peek(NONE).begin <= now:
            note = next(self.waits)
            heappush(self.notes, note)

        width, height = self.size

        image_np = np.array(self.image)

        surface = gizeh.Surface(width, height)

        gizeh_image = gizeh.ImagePattern(image_np, pixel_zero=(width / 2, height / 2))
        gizeh.rectangle(lx=width, ly=height, xy=(width / 2, height / 2), fill=gizeh_image).draw(surface)


        for note in self.notes:
            rect = self.point_finger(note, now)
            rect.draw(surface)
            check_result = self.check(note)
            acc = self.printacc(check_result)
            acc.draw(surface)
            circle = self.predict(note, now)
            if circle is not None:
                circle.draw(surface)

        return surface.get_npimage()
    
    def point_finger(self, note, now):
        if note:
            track = self.midi.notes[note.index]['track']
            track12 = self.midi.notes[note.index]['track12']
            switch = self.midi.notes[note.index]['switch']

        center = POSITION[int(track) % 10]
        if (track != track12):
            if now > switch:
                center = POSITION[int(track12) % 10]
                
        return gizeh.text("○", fontfamily="Consolas", fontsize=50, fill=(1, 1, 1), xy=center, stroke=(0, 0, 0))
    
    def predict(self, note, now):
        if note:
            track = self.midi.notes[note.index]['track']
            track12 = self.midi.notes[note.index]['track12']
            track2 = self.midi.notes[note.index]['track2']
            track22 = self.midi.notes[note.index]['track22']
            switch = self.midi.notes[note.index]['switch']

        record = None
        if now > switch:
            if(track12 != track22):
                record = track22
        else:
            if(track != track2):
                record = track2
        if record is not None:
            center = POSITION[int(record) % 10]
            adjusted_center = (center[0], center[1] - 35)
            return gizeh.text("X", fontfamily="Consolas", fontsize=50, fill=(1, 0, 0), xy=adjusted_center, stroke=(0, 0, 0))
        return None
    
    def check(self, note):
        if note :
            track = self.midi.notes[note.index]['track']
            track12 = self.midi.notes[note.index]['track12']
            track2 = self.midi.notes[note.index]['track2']
            track22 = self.midi.notes[note.index]['track22']
        
        if self.checknum != note.index :
            self.checknum = note.index
            if (track == track2 and track12 == track22) and note.index not in self.checked_notes:
                self.checked_notes.add(note.index)
                return True
            else :
                self.checked_notes.add(note.index)

    
    def printacc(self, check_result):
        color = (1, 1, 1)
        if check_result:  # 使用在 check 函數中設置的布林值
            self.correct += 1
            color = (1, 1, 1)
        else :
            color = (1, 0 ,0)

        # 在繪製新的文字之前，先繪製一個與背景顏色相同的矩形來覆蓋舊的文字
        bg_rect = gizeh.rectangle(lx=200, ly=50, xy=(330,358), fill=(0, 0, 0))

        acc = self.correct / self.total if self.total > 0 else 0
        acc_str = 'Accuracy: {:.2f}%'.format(acc * 100)
        text =  gizeh.text(acc_str, fontfamily="Arial", fontsize=50, fill=color, xy=(330,358), stroke=(0, 0, 0))

        return gizeh.Group([bg_rect, text])
    
def midi_videoclip(midi, size=(1920, 1080)): # draw frames, upper: notes, lower: keyboards, hand: hand
    lower_size = (size[0], int(size[0] / 52 * 6))
    upper_size = (size[0], size[1] - lower_size[1])
    lower_part = PianoPart(midi, lower_size)
    upper_part = ForeseePart(midi, upper_size)

    image_path = './Utils./image.png'
    image = Image.open(image_path)
    hand_part = HandPart(midi, image)

    duration = midi.get_duration()  # expensive call
    upper_clip = mpy.VideoClip(upper_part.make_frame, duration=duration)
    lower_clip = mpy.VideoClip(lower_part.make_frame, duration=duration).set_position(("center", upper_size[1]))
    hand_clip = mpy.VideoClip(hand_part.make_frame, duration=duration).set_position((1300, 400)).set_opacity(0.7)

    # Composite the clips
    final_clip = mpy.CompositeVideoClip([upper_clip, lower_clip, hand_clip], size=size)

    return final_clip

if __name__ == '__main__':
    from Parser import Midi
    root = tk.Tk()
    root.withdraw()
    midi_file_path = filedialog.askopenfilename()
    comparison_files = filedialog.askopenfilenames()
    filename = midi_file_path.rsplit('.', 1)[0]
    midi = Midi(filename, comparison_files)
    clip = midi_videoclip(midi)
    audio_clip = mpy.AudioFileClip(f'{filename}.wav').set_duration(clip.duration)
    clip_with_audio = clip.set_audio(audio_clip)
    clip_with_audio.write_videofile(f'{filename}.mp4', fps=30, audio=True)