import os
import csv
import json
import math
import queue
import pandas
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message
import music21
from music21 import pitch
import Utils.KeyPos as KP

class NoteData:
    def __init__(self):
        self.note_number = None
        self.onset_time_in_sec = None
        self.offset_time_in_sec = None
        self.duration_in_sec = None
        self.pitch = None
        self.keypos_x = None
        self.keypos_y = None
        self.onset_velocity = None
        self.offset_velocity = None

class TimeSignature: # not use
    def __init__(self, msg, time):
        self.numerator = msg.numerator
        self.denominator = msg.denominator
        self.clocks_per_click = msg.clocks_per_click
        self.notated_32nd_notes_per_beat = msg.notated_32nd_notes_per_beat
        self.abs_time_in_sec = time

class Tempo:
    def __init__(self, tempo, time):
        self.tempo = tempo
        self.abs_time_in_sec = time

class Parser(mido.MidiFile):
    def __init__(self, filename):
        try:
            self.midifile = mido.MidiFile(f'./MidiData./MidiFiles./{filename}.mid')
        except:
            print(f'File {filename}.mid not found!')
            exit()
        self.filename = filename
        self.pianoGM = ['Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano', 'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clavinet']
    
    def judgeInsrtumentType(self):
        if len(self.midifile.tracks) != 2:
            print('Illegal format!')
            print('Required format in Musescore 3:')
            print('track 0 for right hand side')
            print('track 1 for left hand side')
            exit(0)
        GM = []
        for msg in self.midifile:
            if msg.type == 'program_change':
                if msg.program > 7:
                    print(f'Input instrument GM {msg.program} not belongs to piano!')
                    exit()
                elif GM:
                    if msg.program not in GM:
                        print('Piano type more than 1!')
                        exit()
                else:
                    GM.append(msg.program)
    
    def tick2second(self, tick, ticks_per_beat, tempo):
        if tempo == None:
            return 0
        return (tick / ticks_per_beat)  * tempo * 1e-6

    def writeMidiToTxt(self):
        with open(f'{self.filename}.txt', 'w') as f:
            for i, track in enumerate(self.midifile.tracks):
                total_time = 0
                f.write(f"Track {i}:\n")
                for msg in track:
                    total_time += msg.time
                    f.write(str(msg) + f', total_time: {total_time}' + '\n')
                f.write('\n')
        
    def getEvents(self):
        tracks = [] # to store final results
        note_on_timeline = [] # to store abs onset times in second
        # save time changes which will affect all tracks
        time_signatures = []
        tempos = []
        for i, track in enumerate(self.midifile.tracks):
            #cur_note_on_timeline = []
            cur_notes =[]
            note_q = []
            pitch_q = []
            cur_time_signature = None # not use
            cur_tempo = None
            # for case that more than 1 track
            ts_index = 0
            tp_index = 0
            abs_time_in_sec = 0
            
            for j, msg in enumerate(track):

                if i != 0:
                    if j == 0: # first msg
                        cur_time_signature = time_signatures[ts_index] # not use
                        cur_tempo = tempos[tp_index].tempo
                    else:
                        while (tp_index+1) < len(tempos) and (abs_time_in_sec + self.tick2second(msg.time, self.midifile.ticks_per_beat, cur_tempo)) >= tempos[tp_index+1].abs_time_in_sec: # tempo change in measure
                            msg.time -= (tempos[tp_index+1].abs_time_in_sec - abs_time_in_sec) * self.midifile.ticks_per_beat / (cur_tempo * 1e-6)
                            abs_time_in_sec = tempos[tp_index+1].abs_time_in_sec
                            tp_index += 1
                            cur_tempo = tempos[tp_index].tempo
                
                abs_time_in_sec = abs_time_in_sec + self.tick2second(msg.time, self.midifile.ticks_per_beat, cur_tempo)

                # record features
                if msg.type == 'time_signature': # not use
                    cur_time_signature = TimeSignature(msg, abs_time_in_sec)
                    # save time signature record
                    time_signatures.append(cur_time_signature)
                elif msg.type == 'set_tempo':
                    cur_tempo = msg.tempo
                    # save tempo record
                    temp_tempo = Tempo(cur_tempo, abs_time_in_sec)
                    tempos.append(temp_tempo)
                elif msg.type == 'note_on' or msg.type == 'note_off': # note events
                    # set note on or note off
                    if msg.type == 'note_on' and msg.velocity != 0: # only use note_on
                        # create note
                        note = NoteData()
                        # set abs onset time
                        note.onset_time_in_sec = round(abs_time_in_sec, 6)
                        # set other infos
                        note.pitch = float(msg.note)
                        note.onset_velocity = float(msg.velocity)
                        keypos = KP.PitchToKeyPos(int(msg.note))
                        note.keypos_x = keypos.x
                        note.keypos_y = keypos.y
                        #add into onset timeline
                        if note.onset_time_in_sec not in note_on_timeline:
                            note_on_timeline.append(note.onset_time_in_sec)
                        # enqueue
                        note_q.append(note)
                        pitch_q.append(note.pitch)
                    else: # use note_on or note_off
                        # find note in queue
                        index = pitch_q.index(msg.note)
                        # dequeue
                        pitch_q.remove(msg.note)
                        note = note_q.pop(index)
                        # set abs offset time
                        note.offset_time_in_sec = round(abs_time_in_sec, 6)
                        # set duration
                        note.duration_in_sec = round(note.offset_time_in_sec - note.onset_time_in_sec, 6)
                        # set offset velocity
                        note.offset_velocity = msg.velocity
                        # add into notes
                        cur_notes.append(note)
                # else temporarily useless
            tracks.append(cur_notes)
        
        # sort onset times
        note_on_timeline = sorted(note_on_timeline)
        
        # write note messages with precise time # just for debug
        with open(f'{self.filename}_track_records.txt', 'w') as f:
            for i, track in enumerate(tracks):
                f.write(f'Track {i}:\n')
                for note in track:
                    f.write(f'note = {note.pitch} on_time = {note.onset_time_in_sec} off_time = {note.offset_time_in_sec}\n')
                f.write('\n')
        
        # write features file
        if len(tracks) == 2:
            right_notes = sorted(tracks[0], key = lambda note: (note.onset_time_in_sec, note.pitch))
            left_notes = sorted(tracks[1], key = lambda note: (note.onset_time_in_sec, note.pitch))
            right_index = 0
            left_index = 0
            note_number = 0
            for time in note_on_timeline:
                right_notes_with_same_onset_time = []
                left_notes_with_same_onset_time = []
                while right_index < len(right_notes) and right_notes[right_index].onset_time_in_sec == time:
                    right_notes_with_same_onset_time.append(right_notes[right_index])
                    right_index += 1
                while left_index < len(left_notes) and left_notes[left_index].onset_time_in_sec == time:
                    left_notes_with_same_onset_time.append(left_notes[left_index])
                    left_index += 1
                if len(right_notes_with_same_onset_time) != 0: # if not empty
                    for r_note in right_notes_with_same_onset_time:
                        r_note.note_number = note_number
                        note_number += 1
                if len(left_notes_with_same_onset_time) != 0: # if not empty
                    for l_note in left_notes_with_same_onset_time:
                        l_note.note_number = note_number
                        note_number += 1
            # write rigth and left file to csv
            fieldnames = ['Note_number', 'Onset_time_in_sec', 'Offset_time_in_sec', 'Duration', 'Pitch', 'KeyPos_x', 'KeyPos_y', 'Onset_velocity', 'Offset_velocity']
            with open(f'{self.filename}_left.csv', 'w', newline = '') as left, open(f'{self.filename}_right.csv', 'w', newline = '') as right:
                # write headers
                left_writer = csv.DictWriter(left, fieldnames = fieldnames)
                left_writer.writeheader()
                right_writer = csv.DictWriter(right, fieldnames = fieldnames)
                right_writer.writeheader()
                # write line
                # write notes to left file
                for note in left_notes:
                    left_writer.writerow({
                        'Note_number': note.note_number,
                        'Onset_time_in_sec': note.onset_time_in_sec,
                        'Offset_time_in_sec': note.offset_time_in_sec,
                        'Duration': note.duration_in_sec,
                        'Pitch': note.pitch,
                        'KeyPos_x': note.keypos_x,
                        'KeyPos_y': note.keypos_y,
                        'Onset_velocity': note.onset_velocity,
                        'Offset_velocity': note.offset_velocity
                    })
                
                # write notes to right file
                for note in right_notes:
                    right_writer.writerow({
                        'Note_number': note.note_number,
                        'Onset_time_in_sec': note.onset_time_in_sec,
                        'Offset_time_in_sec': note.offset_time_in_sec,
                        'Duration': note.duration_in_sec,
                        'Pitch': note.pitch,
                        'KeyPos_x': note.keypos_x,
                        'KeyPos_y': note.keypos_y,
                        'Onset_velocity': note.onset_velocity,
                        'Offset_velocity': note.offset_velocity
                    })

def main():
    filename = input("input a midi file name: ")
    parser = Parser(filename)
    parser.judgeInsrtumentType()
    # parser.writeMidiToTxt() # just for debug
    parser.getEvents()

if __name__ == '__main__':
    main()