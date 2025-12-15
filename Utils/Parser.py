import os
import intervaltree
import pandas as pd
from music21 import note

class Note:
    def __init__(self, begin, end, index):
        self.begin = begin
        self.end = end
        self.index = index

    def __lt__(self, other):
        return self.end < other.end

    def __repr__(self):
        return 'Note(%s, %d..%d)' % (self.index, self.begin, self.end)


class Midi():
    def __init__(self, file=None, comparison_files=None):
        self.midi = None
        self.notes = list()
        self.timeline = intervaltree.IntervalTree() # indexed by second intervals
        self.duration = 0
        self.total = 0
        if file:
            self.parse(file, comparison_files)

    def parse(self, midi_file, comparison_files):
        left = pd.read_csv(f'{midi_file}_left.csv')
        right = pd.read_csv(f'{midi_file}_right.csv')

        if comparison_files:
            for file in comparison_files:
                if 'left_true' in file:
                    left_true = pd.read_csv(file)
                elif 'right_true' in file:
                    right_true = pd.read_csv(file)
                elif 'left_pred' in file:
                    left_pred = pd.read_csv(file)
                elif 'right_pred' in file:
                    right_pred = pd.read_csv(file)

        left = pd.concat([left, left_true, left_pred], axis=1)
        right = pd.concat([right, right_true, right_pred], axis=1)

        # 如果存在 left_true 和 right_true, 要合併相關的欄位
        if left_true is not None:
            if 'End_fingering' not in left_true.columns:
                # 如果左邊資料中沒有 End_fingering，則將 Begin_fingering 複製到 End_fingering
                left_true['End_fingering'] = left_true['Begin_fingering']
            left = pd.concat([left, left_true[['Begin_fingering', 'End_fingering']], left_pred], axis=1)

        if right_true is not None:
            if 'End_fingering' not in right_true.columns:
                # 如果右邊資料中沒有 End_fingering，則將 Begin_fingering 複製到 End_fingering
                right_true['End_fingering'] = right_true['Begin_fingering']
            right = pd.concat([right, right_true[['Begin_fingering', 'End_fingering']], right_pred], axis=1)

        merged_df = pd.concat([left, right])
        merged_df.reset_index(drop=True, inplace=True)

        self.total = len(merged_df)

        for index, row in merged_df.iterrows():
            start = float(row.iloc[1])
            end = float(row.iloc[2])

            if start == end:
                # 在編曲時"播放"被設定為"不播放"導致的。
                # 在遇到兩聲部要寫同一個音符時
                # 例如44拍，第一聲部是C4,D4（二分音符）
                # 第二聲部是C4, C4#, D4, D4#（四分音符）
                # 那為了避免同樣的音重複發生，我會把第二聲部的C4和D4的"播放"關閉。
                # 而在MuseScore中，這會使他的start time和end time被設為同一時間。
                # 但如果直接continue會造成shape不同導致error
                # 因此設定一個極短的時間
                print(f'Warning: start == end: {row.iloc[1]} {row.iloc[2]}')
                end = start + 0.000001

            self.notes.append({'note' : note.Note(int(row.iloc[4])).pitch.midi,
                               'track': int(row.iloc[9]) + 11 if int(row.iloc[9]) < 0 else int(row.iloc[9]),
                               'track12': int(row.iloc[10]) + 11 if int(row.iloc[10]) < 0 else int(row.iloc[10]),
                               'track2': int(row.iloc[11]) + 11 if int(row.iloc[11]) < 0 else int(row.iloc[11]),
                               'track22': int(row.iloc[12]) + 11 if int(row.iloc[12]) < 0 else int(row.iloc[12]),
                               'switch': (start + end) / 2})
            
            self.timeline[start:end] = index

            if self.duration < float(row.iloc[2]):
                self.duration = float(row.iloc[2])

    def get_duration(self):
        return self.duration