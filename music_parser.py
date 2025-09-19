# music_parser_v2.py
import json

VALID_KEYS = {
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
    'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z',
    'x', 'c', 'v', 'b', 'n', 'm',
    '!', '@', '$', '%', '^', '*', '(',
    'Q', 'W', 'E', 'T', 'Y', 'I', 'O', 'P',
    'S', 'D', 'G', 'H', 'J', 'L', 'Z', 'C', 'V', 'B'
}

def parse_sheet_music(sheet_music_string: str) -> list:
    """Parse with proper understanding of simultaneous notes vs sequences"""
    actions = []
    i = 0
    
    while i < len(sheet_music_string):
        char = sheet_music_string[i]
        
        # Chords: [asdf] = play simultaneously as ONE action
        if char == '[': # <-- This was changed from 'elif' to 'if'
            i += 1
            chord_notes = []
            seen = set()
            while i < len(sheet_music_string) and sheet_music_string[i] != ']':
                ch = sheet_music_string[i]
                if ch in VALID_KEYS and ch not in seen:
                    seen.add(ch)
                    chord_notes.append(ch)
                i += 1

            if chord_notes:
                # Canonical form: sort for order-invariance
                chord_notes = sorted(chord_notes)
                actions.append({
                    'type': 'chord',
                    'notes': chord_notes,
                    'simultaneous': True
                })
            i += 1
            
        # Pauses: | with possible spaces
        elif char == '|':
            duration = 1
            i += 1
            # Count consecutive pauses and spaces
            while i < len(sheet_music_string) and sheet_music_string[i] in [' ', '|']:
                if sheet_music_string[i] == '|':
                    duration += 1
                i += 1
            actions.append({'type': 'pause', 'duration': duration})
            
        # Single notes
        elif char in VALID_KEYS:
            actions.append({'type': 'note', 'note': char})
            i += 1
        else:
            i += 1
    
    return actions
