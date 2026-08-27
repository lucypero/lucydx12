package audio

import "core:time"
import "core:log"

// windows multimedia bindings
HMIDIOUT :: distinct rawptr

foreign import winmm "system:winmm.lib"

@(default_calling_convention="system")
foreign winmm {
	midiOutOpen     :: proc(lphMidiOut: ^HMIDIOUT, uDeviceID: u32, dwCallback: uint, dwInstance: uint, dwFlags: u32) -> u32 ---
	midiOutClose    :: proc(hMidiOut: HMIDIOUT) -> u32 ---
	midiOutShortMsg :: proc(hMidiOut: HMIDIOUT, dwMsg: u32) -> u32 ---
}

// standard 12-tone notes with enharmonic equivalents
Notes :: enum {
	C = 0,
	Cs = 1, Db = 1, 
	D = 2,
	Ds = 3, Eb = 3,
	E = 4,
	F = 5,
	Fs = 6, Gb = 6,
	G = 7,
	Gs = 8, Ab = 8,
	A = 9,
	As = 10, Bb = 10,
	B = 11,
}


g_midi_device: HMIDIOUT

Active_Note :: struct {
	off_msg:  u32,
	end_time: time.Time,
}

g_active_notes: [dynamic]Active_Note


// init windows multimedia device and return true on success
init :: proc() -> bool 
{
	// DeviceID of 0xFFFFFFFF tells windows to route midi to systems default synth
	MIDI_MAPPER : u32 = 0xFFFFFFFF
	res := midiOutOpen(&g_midi_device, MIDI_MAPPER, 0, 0, 0)
	
    if res != 0
    {
        log.warn("Failed to initialize windows multimedia device!")
        return false
    }
	
	g_active_notes = make([dynamic]Active_Note)
	return true
}

destroy :: proc() 
{
	if g_midi_device != nil 
    {
		// stop any currently playing notes and delete note buffer
		for note in g_active_notes 
        {
			midiOutShortMsg(g_midi_device, note.off_msg)
		}
		delete(g_active_notes)
		
		midiOutClose(g_midi_device)
		g_midi_device = nil
	}
}


// call once per frame, kills any note that exceeds its duration
update :: proc() 
{
	if g_midi_device == nil do return

	now := time.now()
	
	// iterate backwards to safely remove
	#reverse for note, i in g_active_notes 
    {
		if time.diff(now, note.end_time) <= 0 
        {
			midiOutShortMsg(g_midi_device, note.off_msg)
			unordered_remove(&g_active_notes, i)
		}
	}
}

// assign midi program instrument to channel
set_instrument :: proc(channel: int, program: int) 
{
	if g_midi_device == nil do return

	ch := u32(clamp(channel, 0 , 15))
	prog := u32(clamp(program, 0, 127))

	// 0xC0 is the Program Change status byte
	msg := u32(0xC0) | ch | (prog << 8)
	midiOutShortMsg(g_midi_device, msg)
}


// play note for duration on a channel
play_note :: proc(note: Notes, octave: int, duration: f32, velocity, channel: int) 
{
	if g_midi_device == nil do return

	// only 16 available channels, so clamp channel to 0-15 range
    ch := u32(clamp(channel, 0, 15))
	
	// calculate midi note from note value and octave and clamp to appropriate range
	raw_note := (octave + 1) * 12 + int(note)
	note_val := u32(max(0, min(127, raw_note)))

	velocity: u32 = u32(velocity) 

	// 0x90 status byte to signal note_on
	status := u32(0x90) | ch
	
    // message structure
    // u32 {(unused byte)(velocity byte)(note byte)(status byte)}
	msg_on  := status  | (note_val << 8) | (velocity << 16)
	msg_off := status | (note_val << 8) | (0 << 16)

	// send note immediately and append to tracker
	if duration > 0 
    {
	    midiOutShortMsg(g_midi_device, msg_on)
		dur := time.Duration(f64(duration) * f64(time.Second))
		append(&g_active_notes, Active_Note{
			off_msg  = msg_off,
			end_time = time.time_add(time.now(), dur),
		})
	}
}