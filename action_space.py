import json
from music_parser import parse_sheet_music, VALID_KEYS

def create_action_mappings(parsed_song: list) -> tuple[dict, list]:
    """
    Creates mappings between musical actions and integer IDs from a parsed song.

    This function identifies all unique actions in a song to create a minimal
    and efficient action space for the RL agent.

    Args:
        parsed_song: A list of action dictionaries from the parse_sheet_music function.

    Returns:
        A tuple containing:
        - action_to_int (dict): A dictionary mapping a string representation of an action to its unique integer ID.
        - int_to_action (list): A list where the index is the ID and the value is the action dictionary.
    """
    unique_actions = []
    
    # Use a set to automatically handle uniqueness
    seen_actions = set()

    for action in parsed_song:
        # Convert the action dictionary to a stable string representation (a tuple of items)
        # so it can be added to a set. We sort the notes in chords to ensure
        # that {'notes': ['a', 's']} and {'notes': ['s', 'a']} are treated as the same action.
        if action['type'] in ['note', 'chord']:
            # Use a tuple of sorted notes to represent the chord/note part
            key_part = tuple(sorted(action['notes']))
            action_key = (action['type'], key_part)
        else: # Pause
            action_key = (action['type'], action['duration'])

        if action_key not in seen_actions:
            seen_actions.add(action_key)
            unique_actions.append(action)

    # Create the forward and reverse mappings
    int_to_action = unique_actions
    action_to_int = {json.dumps(action): i for i, action in enumerate(int_to_action)}
    
    return action_to_int, int_to_action

# --- DEMONSTRATION ---
if __name__ == "__main__":
    # The same complex example song string from the parser
    test_song = """
    [yiu] [yiu] o [yiu] | [yiu] i
    [yiu] u [yiu] | | [yiu] u y T
    [as] [sa] o
    """

    print("--- 1. Parsing Song ---")
    parsed_song = parse_sheet_music(test_song)
    print(f"Song has {len(parsed_song)} total actions (including repeats).")

    print("\n--- 2. Creating Action Mappings ---")
    action_to_int, int_to_action = create_action_mappings(parsed_song)
    
    print(f"Found {len(int_to_action)} unique actions.")
    print("This number is the size of our action space for the neural network.")

    print("\n--- Mapping from Integer to Action (int_to_action) ---")
    for i, action in enumerate(int_to_action):
        print(f"{i}: {action}")
        
    print("\n--- Example Lookup: What is action ID 4? ---")
    action_id = 4
    if action_id < len(int_to_action):
      print(f"ID {action_id} corresponds to the action: {int_to_action[action_id]}")

    print("\n--- Example Lookup: What is the ID for the note 'o'? ---")
    note_o_action = {'type': 'note', 'notes': ['o']}
    # We must use json.dumps to look it up in the action_to_int map
    note_o_id = action_to_int.get(json.dumps(note_o_action))
    if note_o_id is not None:
        print(f"The action {note_o_action} has the ID: {note_o_id}")
