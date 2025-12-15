import math

class KeyPos:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

def SubtrKeyPos(kp1, kp2):
    # interval from kp2 to kp1
    keypos = KeyPos()
    keypos.x = kp1.x - kp2.x
    keypos.y = kp1.y - kp2.y
    return keypos

def AddKeyPos(kp1, kp2):
    keypos = KeyPos()
    keypos.x = kp1.x + kp2.x
    keypos.y = kp1.y + kp2.y
    return keypos

def PitchToKeyPos(pitch):
    # (root height, white(0) or black(1) key)
    # C4 = 60 -> (0, 0), Eb4 = 63 -> (1, 1)
    pc = int(pitch % 12)
    octave = int(pitch / 12 - 1)
    
    keypos = KeyPos()
    if pc in [0, 1]: # C, C#
        keypos.x = 0
    elif pc in [2, 3]: # D, D#
        keypos.x = 1
    elif pc == 4: # E
        keypos.x = 2
    elif pc in [5, 6]: # F, F#
        keypos.x = 3
    elif pc in [7, 8]: # G, G#
        keypos.x = 4
    elif pc in [9, 10]: # A, A#
        keypos.x = 5
    elif pc == 11: # B
        keypos.x = 6
    # else error

    keypos.x += 7 * (octave - 4)
    if pc in [0, 2, 4, 5, 7, 9, 11]: # white key
        keypos.y = 0
    elif pc in [1, 3, 6, 8, 10]: # black key
        keypos.y = 1
    # else error

    return keypos

def KeyPosToPitch(keypos): # ex. (3, 0) -> 65 (F4)
    if keypos.x + 70 < 0 or keypos.y not in [0, 1]: # error
        print(f'undefined keyPos! ({keypos.x}, {keypos.y})')
        assert False
    octave = int(math.floor((keypos.x + 70) / 7 - 6))
    xmod7 = int((keypos.x + 70) % 7)
    if xmod7 == 0:
        pc = 0
    elif xmod7 == 1:
        pc = 2
    elif xmod7 == 2:
        pc = 4
    elif xmod7 == 3:
        pc = 5
    elif xmod7 == 4:
        pc = 7
    elif xmod7 == 5:
        pc = 9
    elif xmod7 == 6:
        pc == 11
    pc += keypos.y
    return 12 * (octave - 4) + pc + 60

# A0: 21 -> (-23, 0)
# C8: 108 -> (28, 0)

if __name__ == "__main__": # test
    kp = PitchToKeyPos(21) # midi number
    print(kp.x)
    print(kp.y)
    kp = PitchToKeyPos(108) # midi number
    print(kp.x)
    print(kp.y)