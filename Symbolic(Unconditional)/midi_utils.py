import mido

#
def midi_to_note_duration_sequence(filename):
    """Extract (pitch, duration) tuples from a MIDI file."""
    mid = mido.MidiFile(filename)
    notes = []
    abs_time = 0
    note_on_times = {}
    for msg in mid:
        abs_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            note_on_times[msg.note] = abs_time
        elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in note_on_times:
            duration = abs_time - note_on_times[msg.note]
            notes.append((msg.note, int(duration * 480)))  # scale duration for MIDI ticks
            del note_on_times[msg.note]
    return notes

def note_sequence_to_midi(note_sequence, filename, velocity=64, tempo=500000, note_length=120):
    import mido
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    for note in note_sequence:
        track.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
        track.append(mido.Message('note_off', note=note, velocity=velocity, time=note_length))
    mid.save(filename)

def note_duration_sequence_to_midi(note_duration_sequence, filename, velocity=64, tempo=500000):

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    for note, duration in note_duration_sequence:
        track.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
        track.append(mido.Message('note_off', note=note, velocity=velocity, time=duration))
    mid.save(filename)