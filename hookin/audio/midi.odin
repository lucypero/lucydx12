package audio

import "core:time"
import "core:os"
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
	Cs = 1,  Db = 1, 
	D = 2,
	Ds = 3,  Eb = 3,
	E = 4,
	F = 5,
	Fs = 6,  Gb = 6,
	G = 7,
	Gs = 8,  Ab = 8,
	A = 9,
	As = 10, Bb = 10,
	B = 11,
}

@(private)
g_midi_device: HMIDIOUT

@(private)
Active_Note :: struct {
	off_msg:  u32,
	end_time: time.Time,
}

@(private)
g_active_notes: [dynamic]Active_Note


// init windows multimedia device and return true on success
@(require_results)
init :: proc() -> (success: bool) {
	success = true
	// DeviceID of 0xFFFFFFFF tells windows to route midi to systems default synth
	MIDI_MAPPER : u32 = 0xFFFFFFFF
	res := midiOutOpen(&g_midi_device, MIDI_MAPPER, 0, 0, 0)

	if res != 0 {
		log.warn("Failed to initialize windows multimedia device!")
		success = false
		return
	}
	g_active_notes = make([dynamic]Active_Note)
	return
}

destroy :: proc() {
	if g_midi_device != nil {
		// stop any currently playing notes and delete note buffer
		for note in g_active_notes {
			midiOutShortMsg(g_midi_device, note.off_msg)
		}
		delete(g_active_notes)

		if g_playback.tracks != nil do delete(g_playback.tracks)

		midiOutClose(g_midi_device)
		g_midi_device = nil
	}
}


// call once per frame, kills any note that exceeds its duration
update :: proc() {
	if g_midi_device == nil do return

	now := time.now()

	// process single note events (play_note)
	// iterate backwards to safely remove
	#reverse for note, i in g_active_notes {
		if time.diff(now, note.end_time) <= 0 {
			midiOutShortMsg(g_midi_device, note.off_msg)
			unordered_remove(&g_active_notes, i)
		}
	}

	// process file playback
	if g_playback.is_playing && g_playback.file != nil {
		dt_sec := time.duration_seconds(time.diff(g_playback.last_time, now))
		g_playback.last_time = now

		// calculate frametime and convert to midi ticks
		sec_per_qn := f64(g_playback.tempo_us_per_qn) / 1_000_000.0
		sec_per_tick := sec_per_qn / f64(g_playback.file.division)
		ticks_to_process := dt_sec / sec_per_tick

		all_finished := true

		// loop through all tracks, skipping finished tracks and subtracting
		// time from active track timers
		for track_idx in 0..<len(g_playback.tracks) {
			tp := &g_playback.tracks[track_idx]
			if tp.finished do continue

			all_finished = false
			tp.tick_timer -= ticks_to_process
			track := &g_playback.file.tracks[track_idx]

			// loop all events that have expired timers (this could be multiple events
			// for things like chords)
			for tp.tick_timer <= 0 && !tp.finished {
				ev := &track.events[tp.event_idx]

				if ev.type == .Midi {
					// for standard midi note, construct the u32 data and fire the message
					msg := u32(ev.status) | (u32(ev.data1) << 8) | (u32(ev.data2) << 16)
					midiOutShortMsg(g_midi_device, msg)
				} else if ev.type == .Meta {
					// if a set tempo (0x51) meta event, pack, calculate and set new tempo
					if ev.meta_type == 0x51 && len(ev.meta_data) == 3 {
						g_playback.tempo_us_per_qn = (u32(ev.meta_data[0]) << 16) | 
						(u32(ev.meta_data[1]) << 8) | 
						u32(ev.meta_data[2])
					}
				}

				// advance to next event, adding next event delay to timer
				// if no more events, mark finished
				tp.event_idx += 1
				if tp.event_idx >= len(track.events) {
					tp.finished = true
				} else {
					next_ev := track.events[tp.event_idx]
					tp.tick_timer += f64(next_ev.delta_ticks)
				}
			}
		}

		if all_finished {
			g_playback.is_playing = false
		}
	}
}

// assign midi program instrument to channel
set_instrument :: proc(channel: int, program: int) {
	if g_midi_device == nil do return

	ch := u32(clamp(channel, 0 , 15))
	prog := u32(clamp(program, 0, 127))

	// 0xC0 is the Program Change status byte
	msg := u32(0xC0) | ch | (prog << 8)
	midiOutShortMsg(g_midi_device, msg)
}


// play note for duration on a channel
play_note :: proc(note: Notes, octave: int, duration: f32, velocity, channel: int) {
	if g_midi_device == nil do return

	// only 16 available channels, so clamp channel to 0-15 range
	ch := u32(clamp(channel, 0, 15))

	// calculate midi note from note value and octave and clamp to appropriate range
	raw_note := (octave + 1) * 12 + int(note)
	note_val := u32(max(0, min(127, raw_note)))

	velocity: u32 = u32(velocity) 

	// 0x90 status byte to signal note_on, 0x80 for note_off
	status_on  := u32(0x90) | ch
	status_off := u32(0x80) | ch

	// message structure
	// u32 {(unused byte)(velocity byte)(note byte)(status byte)}
	msg_on  := status_on  | (note_val << 8) | (velocity << 16)
	msg_off := status_off | (note_val << 8) | (64 << 16)

	// send note immediately and append to tracker
	if duration > 0 {
		midiOutShortMsg(g_midi_device, msg_on)
		dur := time.Duration(f64(duration) * f64(time.Second))
		append(&g_active_notes, Active_Note{
			off_msg  = msg_off,
			end_time = time.time_add(time.now(), dur),
		})
	}
}

// midi data structures

MidiEventType :: enum {
	Midi, 	// standard note data and pitch bends
	Meta, 	// tempo changes, track names, etc
	SysEx, 	// custom hardware instructions
}

MidiEvent :: struct {
	delta_ticks: u32,			// time delay since last event
	type:        MidiEventType,	
	status:      u8,			// status byte for things like note on and off messages
	data1:       u8,			// data bytes for things like note and velocity (depends on the status byte)
	data2:       u8,
	meta_type:   u8,			// identifier for meta events (like tempo change)
	meta_data:   []u8, 
}

MidiTrack :: struct {
	events: [dynamic]MidiEvent,	// array of sequenced midi events
}

MidiFile :: struct {
	raw_data: []u8,			// raw .mid file bytes
	format:   u16, 			// format info (one track, multiple tracks, etc)
	division: u16, 			// ticks per quarter note
	tracks:   []MidiTrack,	// the various event tracks that belong to the .mid file
}

// playback state

TrackPlayback :: struct {
	event_idx:  int,
	tick_timer: f64, // how many ticks till next event fires
	finished:   bool,
}

Midi_Playback_State :: struct {
	file:            ^MidiFile,
	is_playing:      bool,
	tempo_us_per_qn: u32, 				// tempo in microseconds per quarter-note
	tracks:          []TrackPlayback,
	last_time:       time.Time,
}

@(private)
g_playback: Midi_Playback_State

// binary parsing helpers

// midi uses big endian, helpers return values in little endian so we can work with them easier
@(private)
read_u16_be :: proc(data: []u8, offset: ^int) -> u16 {
	val := (u16(data[offset^]) << 8) | u16(data[offset^ + 1])
	offset^ += 2
	return val
}

@(private)
read_u32_be :: proc(data: []u8, offset: ^int) -> u32 {
	val := (u32(data[offset^]) << 24) | (u32(data[offset^ + 1]) << 16) | (u32(data[offset^ + 2]) << 8) | u32(data[offset^ + 3])
	offset^ += 4
	return val
}

// midi uses variable length quantity to save space for time delays
// bottom 7 bits contain the data, top bit is a flag
// if flag is 0 then this is the last byte
@(private)
read_vlq :: proc(data: []u8, offset: ^int) -> u32 {
	val: u32 = 0
	for {
		b := data[offset^]
		offset^ += 1
		val = (val << 7) | u32(b & 0x7F)
		if (b & 0x80) == 0 do break
	}
	return val
}

// midi loading and playback code

load_midi_file :: proc(filepath: string) -> (file: MidiFile, success: bool) {
	success = true
	data, err := os.read_entire_file_from_path(filepath, context.allocator)
	if err != .SUCCESS {
		log.warn("Failed to read file: ", filepath)
		success = false
		return
	}

	file.raw_data = data
	defer if !success do destroy_midi_file(&file)
	offset := 0

	// standard midi header
	// 4 bytes -> MThd string
	// 4 bytes -> Header Length
	// 2 bytes -> Format Data
	// 2 bytes -> Track Count
	// 2 bytes -> Division
	// potential padding
	// start of first track MTrk chunk

	// check if file is large enough to contain header and look for 'MThd' string to validate as .mid file
	if offset + 14 > len(data) || data[0] != 'M' || data[1] != 'T' || data[2] != 'h' || data[3] != 'd' {
		log.warn("Failed to validate midi file: ", filepath)
		success = false
		return
	}
	offset += 4

	header_len := read_u32_be(data, &offset)
	file.format = read_u16_be(data, &offset)
	track_count := read_u16_be(data, &offset)
	file.division = read_u16_be(data, &offset)

	// skip unused header bytes, account for potentially non-standard header size
	offset += int(header_len) - 6

	file.tracks = make([]MidiTrack, track_count)

	for i in 0..<int(track_count) {
		// standard track chunk
		// 4 bytes -> MTrk string
		// 4 bytes -> Track Length
		// Track Events until Track Length in bytes


		// check length and validate MTrk chunk
		if offset + 8 > len(data) || data[offset] != 'M' || data[offset+1] != 'T' || data[offset+2] != 'r' || data[offset+3] != 'k' {
			log.warn("Failed to parse MTrk chunk, invalid or corrupted file!")
			success = false
			return
		}
		offset += 4

		track_len := int(read_u32_be(data, &offset))
		end_offset := offset + track_len

		track := &file.tracks[i]
		track.events = make([dynamic]MidiEvent)

		// midi compression trick, if status is omitted the device applies the previously used byte
		running_status: u8 = 0

		// parse events until end of track block
		for offset < end_offset {
			ev := MidiEvent{}
			ev.delta_ticks = read_vlq(data, &offset)

			// status bytes are above 0x80, if current value is above 0x80
			// we know its a new status byte and updated the running status
			// if its below 0x80 it is data and we reuse the last running
			// status and skip the offset to treat it as data
			status_byte := data[offset]
			if status_byte >= 0x80 {
				running_status = status_byte
				offset += 1
			}
			ev.status = running_status

			// handle meta events (marked as 0xFF)
			if running_status == 0xFF {
				ev.type = .Meta
				ev.meta_type = data[offset]
				offset += 1
				length := int(read_vlq(data, &offset))
				ev.meta_data = data[offset : offset+length]
				offset += length
				// handle SysEx events
			} else if running_status == 0xF0 || running_status == 0xF7 {
				ev.type = .SysEx
				length := int(read_vlq(data, &offset))
				ev.meta_data = data[offset : offset+length]
				offset += length
				// handle standard midi events
			} else {
				ev.type = .Midi
				// the bottom four bits of status byte are channel information
				// extract just the top four bits to check the core message type
				msg_type := running_status & 0xF0
				ev.data1 = data[offset]
				offset += 1

				// Program Change (0xC0) and Aftertouch (0XD0) have 1 byte, all other events
				// have 2 data bytes
				if msg_type != 0xC0 && msg_type != 0xD0 {
					ev.data2 = data[offset]
					offset += 1
				}
			}

			append(&track.events, ev)
		}
	}
	return
}

play_midi :: proc(file: ^MidiFile) {
	// clear tracks if something was already playing
	if g_playback.tracks != nil {
		delete(g_playback.tracks)
	}

	g_playback.file = file
	g_playback.is_playing = true
	g_playback.tempo_us_per_qn = 500000 // 500000qn == 120bpm, set to default tempo
	g_playback.last_time = time.now()
	g_playback.tracks = make([]TrackPlayback, len(file.tracks))

	// set timer of every track to the delay time of first event, empty tracks get marked finished
	for track, i in file.tracks {
		if len(track.events) > 0 {
			g_playback.tracks[i].tick_timer = f64(track.events[0].delta_ticks)
		} else {
			g_playback.tracks[i].finished = true
		}
	}
}

stop_midi :: proc() {
	g_playback.is_playing = false
	if g_midi_device == nil do return

	// set note off messages to every channel to avoid hanging notes
	for ch in 0..<16 {
		msg := u32(0xB0) | u32(ch) | (123 << 8) | (0 << 16)
		midiOutShortMsg(g_midi_device, msg)
	}
}

destroy_midi_file :: proc(file: ^MidiFile) {
	for track in file.tracks {
		delete(track.events)
	}
	delete(file.tracks)
	delete(file.raw_data)
	file.raw_data = nil
}

