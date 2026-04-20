"""
Serum Brother Wavetable Engine
Professional-grade offline wavetable synthesis engine
Inspired by Xfer Records Serum
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable, Any
from enum import Enum, IntEnum
import json
import os


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class WarpMode(Enum):
    BEND_PLUS = "bend_plus"
    BEND_MINUS = "bend_minus"
    PWM = "pwm"
    FM = "fm"
    NONE = "none"


class LFOShape(Enum):
    SINE = "sine"
    TRIANGLE = "triangle"
    SAW = "saw"
    SQUARE = "square"
    SAMPLE_HOLD = "sample_hold"


class LFOMode(Enum):
    FREE = "free_running"
    SYNC = "sync"


class WaveType(IntEnum):
    SINE = 1
    SQUARE = 2
    SAW = 3
    TRIANGLE = 4
    PULSE_PWM = 5
    NOISE = 6
    CUSTOM = 7


# ============================================================================
# WAVETABLE ENGINE
# ============================================================================

@dataclass
class WavetableFrame:
    data: np.ndarray
    normalize: bool = True

    def __post_init__(self):
        if self.normalize and len(self.data) > 0:
            max_val = np.max(np.abs(self.data))
            if max_val > 0:
                self.data = self.data / max_val


class Wavetable:
    def __init__(self, table_size: int = 2048, num_frames: int = 256):
        self.table_size = table_size
        self.num_frames = num_frames
        self.frames: List[WavetableFrame] = []
        self._init_basic_wavetable()

    def _init_basic_wavetable(self):
        t = np.linspace(0, 2 * np.pi, self.table_size, endpoint=False)

        # Basic waveforms
        sine = np.sin(t)
        triangle = 2 * np.arcsin(np.sin(t)) / np.pi
        saw = 2 * ((t / (2 * np.pi)) % 1) - 1
        square = np.where(t < np.pi, 1.0, -1.0)

        # Generate frames with morphing
        for i in range(self.num_frames):
            pos = i / (self.num_frames - 1)

            if pos < 0.25:  # Sine to Triangle
                blend = pos * 4
                frame = (1 - blend) * sine + blend * triangle
            elif pos < 0.5:  # Triangle to Saw
                blend = (pos - 0.25) * 4
                frame = (1 - blend) * triangle + blend * saw
            elif pos < 0.75:  # Saw to Square
                blend = (pos - 0.5) * 4
                frame = (1 - blend) * saw + blend * square
            else:  # Square to narrow pulse
                blend = (pos - 0.75) * 4
                pulse_width = 0.5 - 0.4 * blend
                frame = np.where(t < (2 * np.pi * pulse_width), 1.0, -1.0)

            self.frames.append(WavetableFrame(frame))

    def get_frame(self, index: float, warp_mode: WarpMode = WarpMode.NONE,
                  warp_amount: float = 0.0) -> np.ndarray:
        if not self.frames:
            return np.zeros(self.table_size)

        index = np.clip(index, 0, len(self.frames) - 1.001)
        idx_low = int(np.floor(index))
        idx_high = min(idx_low + 1, len(self.frames) - 1)
        frac = index - idx_low

        frame = (1 - frac) * self.frames[idx_low].data + frac * self.frames[idx_high].data

        if warp_mode != WarpMode.NONE:
            frame = self._apply_warp(frame, warp_mode, warp_amount)

        return frame

    def _apply_warp(self, frame: np.ndarray, mode: WarpMode, amount: float) -> np.ndarray:
        n = len(frame)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)

        if mode == WarpMode.BEND_PLUS:
            warp_factor = 1 + amount * 3
            phase = np.power(t / (2 * np.pi), warp_factor) * 2 * np.pi
            indices = (phase / (2 * np.pi) * n).astype(int) % n
            return frame[indices]
        elif mode == WarpMode.BEND_MINUS:
            warp_factor = 1 + amount * 3
            phase = (1 - np.power(1 - t / (2 * np.pi), warp_factor)) * 2 * np.pi
            indices = (phase / (2 * np.pi) * n).astype(int) % n
            return frame[indices]
        elif mode == WarpMode.PWM:
            width = 0.5 + amount * 0.4
            return np.where(t < (2 * np.pi * width), np.max(frame), np.min(frame))
        elif mode == WarpMode.FM:
            modulator = np.sin(t * (1 + amount * 5))
            phase_shift = modulator * amount * np.pi
            shifted_t = (t + phase_shift) % (2 * np.pi)
            indices = (shifted_t / (2 * np.pi) * n).astype(int) % n
            return frame[indices]

        return frame


class WavetableOscillator:
    def __init__(self, wavetable: Wavetable, sample_rate: float = 44100):
        self.wavetable = wavetable
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.phase_increment = 0.0
        self.wt_position = 0.0
        self.warp_mode = WarpMode.NONE
        self.warp_amount = 0.0

    def set_frequency(self, freq: float):
        self.phase_increment = freq / self.sample_rate

    def process_block(self, num_samples: int) -> np.ndarray:
        output = np.zeros(num_samples)
        phase_delta = np.full(num_samples, self.phase_increment)

        for i in range(num_samples):
            frame = self.wavetable.get_frame(self.wt_position, self.warp_mode, self.warp_amount)
            idx = self.phase * self.wavetable.table_size
            idx_low = int(np.floor(idx))
            idx_high = (idx_low + 1) % self.wavetable.table_size
            frac = idx - idx_low

            output[i] = (1 - frac) * frame[idx_low] + frac * frame[idx_high]
            self.phase += phase_delta[i]
            self.phase %= 1.0

        return output


# ============================================================================
# OSCILLATOR SYSTEM
# ============================================================================

@dataclass
class OscillatorConfig:
    wave_type: int = 1
    level: float = 1.0
    pan: float = 0.0
    coarse_pitch: float = 0.0
    fine_pitch: float = 0.0
    unison_voices: int = 1
    unison_detune: float = 0.0
    unison_spread: float = 0.0
    wt_position: float = 0.0
    warp_mode: WarpMode = WarpMode.NONE
    warp_amount: float = 0.0
    enabled: bool = True


class BasicOscillator:
    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.phase_increment = 0.0
        self.pulse_width = 0.5

    def set_frequency(self, freq: float):
        self.phase_increment = freq / self.sample_rate

    def generate(self, wave_type: int, num_samples: int) -> np.ndarray:
        phases = (self.phase + np.arange(num_samples) * self.phase_increment) % 1.0
        self.phase = (self.phase + self.phase_increment * num_samples) % 1.0

        if wave_type == WaveType.SINE:
            return np.sin(phases * 2 * np.pi)
        elif wave_type == WaveType.SQUARE:
            return np.where(phases < self.pulse_width, 1.0, -1.0)
        elif wave_type == WaveType.SAW:
            return 2 * phases - 1
        elif wave_type == WaveType.TRIANGLE:
            return 2 * np.abs(2 * phases - 1) - 1
        elif wave_type == WaveType.NOISE:
            return np.random.uniform(-1, 1, num_samples)
        else:
            return np.sin(phases * 2 * np.pi)


class OscillatorBank:
    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.wavetable = Wavetable()
        self.osc1_config = OscillatorConfig(wave_type=1)
        self.osc2_config = OscillatorConfig(wave_type=2)
        self.sub_config = OscillatorConfig(wave_type=1, level=0.5)

        self.osc1_basic = BasicOscillator(sample_rate)
        self.osc2_basic = BasicOscillator(sample_rate)
        self.sub_basic = BasicOscillator(sample_rate)
        self.osc1_wt = WavetableOscillator(self.wavetable, sample_rate)
        self.osc2_wt = WavetableOscillator(self.wavetable, sample_rate)

    def calculate_frequency(self, base_note: float, coarse: float = 0.0, fine: float = 0.0) -> float:
        semitone_offset = coarse + fine / 100.0
        return 440.0 * 2 ** ((base_note - 69 + semitone_offset) / 12.0)

    def process_oscillator(self, config: OscillatorConfig, basic_osc: BasicOscillator,
                          wt_osc: WavetableOscillator, base_freq: float,
                          num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if not config.enabled or config.level <= 0:
            return np.zeros(num_samples), np.zeros(num_samples)

        freq = self.calculate_frequency(base_freq, config.coarse_pitch, config.fine_pitch)

        # Generate audio
        if config.wave_type in [WaveType.CUSTOM, WaveType.PULSE_PWM]:
            wt_osc.set_frequency(freq)
            wt_osc.wt_position = config.wt_position
            wt_osc.warp_mode = config.warp_mode
            wt_osc.warp_amount = config.warp_amount
            mono = wt_osc.process_block(num_samples)
        else:
            basic_osc.set_frequency(freq)
            mono = basic_osc.generate(config.wave_type, num_samples)

        # Apply unison if configured
        if config.unison_voices > 1:
            voices = []
            detune_semitones = config.unison_detune / 100.0
            for i in range(config.unison_voices):
                offset = np.linspace(-detune_semitones, detune_semitones, config.unison_voices)[i]
                voice_freq = freq * (2 ** (offset / 12.0))

                if config.wave_type in [WaveType.CUSTOM, WaveType.PULSE_PWM]:
                    wt_osc.set_frequency(voice_freq)
                    voice = wt_osc.process_block(num_samples)
                else:
                    basic_osc.set_frequency(voice_freq)
                    voice = basic_osc.generate(config.wave_type, num_samples)
                voices.append(voice)

            mono = np.mean(voices, axis=0) * (1.0 / np.sqrt(config.unison_voices))

        # Apply panning and level
        left_gain = config.level * np.cos((config.pan + 1) * np.pi / 4)
        right_gain = config.level * np.sin((config.pan + 1) * np.pi / 4)

        return mono * left_gain, mono * right_gain

    def process_block(self, base_freq: float, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        left_out = np.zeros(num_samples)
        right_out = np.zeros(num_samples)

        # Process main oscillators
        l1, r1 = self.process_oscillator(self.osc1_config, self.osc1_basic,
                                         self.osc1_wt, base_freq, num_samples)
        l2, r2 = self.process_oscillator(self.osc2_config, self.osc2_basic,
                                         self.osc2_wt, base_freq, num_samples)

        left_out = l1 + l2
        right_out = r1 + r2

        # Process sub oscillator
        if self.sub_config.enabled:
            sub_freq = self.calculate_frequency(base_freq - 12)
            self.sub_basic.set_frequency(sub_freq)
            sub_out = self.sub_basic.generate(WaveType.SINE, num_samples)
            left_out += sub_out * self.sub_config.level * 0.707
            right_out += sub_out * self.sub_config.level * 0.707

        return left_out, right_out


# ============================================================================
# MODULATION SYSTEM
# ============================================================================

class Envelope:
    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.attack = 0.01
        self.decay = 0.1
        self.sustain = 0.7
        self.release = 0.5
        self.state = 0.0
        self.stage = 'idle'

    def trigger(self):
        self.stage = 'attack'

    def release_note(self):
        self.stage = 'release'

    def process(self, num_samples: int) -> np.ndarray:
        output = np.zeros(num_samples)

        for i in range(num_samples):
            if self.stage == 'attack':
                self.state += 1.0 / (self.attack * self.sample_rate)
                if self.state >= 1.0:
                    self.state = 1.0
                    self.stage = 'decay'
            elif self.stage == 'decay':
                self.state -= (1.0 - self.sustain) / (self.decay * self.sample_rate)
                if self.state <= self.sustain:
                    self.state = self.sustain
                    self.stage = 'sustain'
            elif self.stage == 'sustain':
                self.state = self.sustain
            elif self.stage == 'release':
                self.state -= self.sustain / (self.release * self.sample_rate)
                if self.state <= 0.0:
                    self.state = 0.0
                    self.stage = 'idle'

            output[i] = self.state

        return output


class LFO:
    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.rate = 1.0
        self.shape = LFOShape.SINE
        self.mode = LFOMode.FREE
        self.phase = 0.0
        self.sync_rate = 1/4

    def process(self, num_samples: int, bpm: float = 120.0) -> np.ndarray:
        rate = (bpm / 60) / self.sync_rate if self.mode == LFOMode.SYNC else self.rate
        phase_increment = rate / self.sample_rate
        phases = (self.phase + np.arange(num_samples) * phase_increment) % 1.0
        self.phase = (self.phase + phase_increment * num_samples) % 1.0

        if self.shape == LFOShape.SINE:
            return np.sin(phases * 2 * np.pi)
        elif self.shape == LFOShape.TRIANGLE:
            return 2 * np.abs(2 * phases - 1) - 1
        elif self.shape == LFOShape.SAW:
            return 2 * phases - 1
        elif self.shape == LFOShape.SQUARE:
            return np.where(phases < 0.5, 1.0, -1.0)
        elif self.shape == LFOShape.SAMPLE_HOLD:
            indices = np.floor(phases * num_samples).astype(int)
            values = np.random.uniform(-1, 1, num_samples)
            return values[indices % num_samples]
        return np.zeros(num_samples)


class ModulationMatrix:
    def __init__(self):
        self.routings: List[Dict] = []
        self.sources: Dict[str, Any] = {}
        self.targets: Dict[str, float] = {}

    def add_routing(self, source: str, target: str, amount: float):
        self.routings.append({'source': source, 'target': target, 'amount': amount})

    def process(self, num_samples: int) -> Dict[str, np.ndarray]:
        mod_values = {target: np.zeros(num_samples) for target in self.targets}

        for routing in self.routings:
            source_val = self.sources.get(routing['source'], 0.0)
            if isinstance(source_val, (int, float)):
                mod_values[routing['target']] += source_val * routing['amount']
            elif isinstance(source_val, np.ndarray):
                mod_values[routing['target']] += source_val * routing['amount']

        return mod_values


# ============================================================================
# FX CHAIN
# ============================================================================

class Distortion:
    def __init__(self, mode: str = 'soft'):
        self.mode = mode
        self.drive = 0.0
        self.mix = 1.0

    def process(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.drive <= 0:
            return left, right

        gain = 10 ** (self.drive / 20)
        left_driven = left * gain
        right_driven = right * gain

        if self.mode == 'soft':
            left_sat = np.tanh(left_driven)
            right_sat = np.tanh(right_driven)
        else:  # hard clip
            left_sat = np.clip(left_driven, -1.0, 1.0)
            right_sat = np.clip(right_driven, -1.0, 1.0)

        left_out = (1 - self.mix) * left + self.mix * left_sat / (gain ** 0.5)
        right_out = (1 - self.mix) * right + self.mix * right_sat / (gain ** 0.5)

        return left_out, right_out


class Phaser:
    """Phaser effect using all-pass filters."""

    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.rate = 0.5
        self.depth = 0.7
        self.feedback = 0.5
        self.mix = 0.5
        self.stages = 6

        # States for each all-pass stage
        self.x1_l = [0.0] * self.stages
        self.y1_l = [0.0] * self.stages
        self.x1_r = [0.0] * self.stages
        self.y1_r = [0.0] * self.stages

        self.lfo_phase = 0.0
        self.f_min = 200
        self.f_max = 2000

    def process(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_out = np.zeros_like(left)
        right_out = np.zeros_like(right)

        for i in range(len(left)):
            # LFO for frequency modulation
            lfo = (np.sin(self.lfo_phase * 2 * np.pi) + 1) / 2
            freq = self.f_min + lfo * self.depth * (self.f_max - self.f_min)

            # Calculate all-pass coefficient
            w0 = 2 * np.pi * freq / self.sample_rate
            alpha = (1 - np.sin(w0)) / np.cos(w0) if abs(np.cos(w0)) > 0.001 else 0

            # Process left channel
            y_l = left[i] + self.feedback * self.y1_l[-1]
            for stage in range(self.stages):
                x_temp = y_l
                y_l = -alpha * x_temp + self.x1_l[stage] + alpha * self.y1_l[stage]
                self.x1_l[stage] = x_temp
                self.y1_l[stage] = y_l

            # Process right channel
            y_r = right[i] + self.feedback * self.y1_r[-1]
            for stage in range(self.stages):
                x_temp = y_r
                y_r = -alpha * x_temp + self.x1_r[stage] + alpha * self.y1_r[stage]
                self.x1_r[stage] = x_temp
                self.y1_r[stage] = y_r

            # Mix dry and wet
            left_out[i] = (1 - self.mix) * left[i] + self.mix * y_l
            right_out[i] = (1 - self.mix) * right[i] + self.mix * y_r

            self.lfo_phase += self.rate / self.sample_rate
            self.lfo_phase %= 1.0

        return left_out, right_out


class Flanger:
    """Flanger effect using modulated delay line."""

    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.rate = 0.2
        self.depth = 0.003
        self.feedback = 0.4
        self.mix = 0.5
        self.delay_ms = 5.0

        # Delay buffer
        max_delay = int(sample_rate * 0.02)  # 20ms max
        self.buffer_l = np.zeros(max_delay)
        self.buffer_r = np.zeros(max_delay)
        self.write_ptr = 0
        self.buffer_size = max_delay

        self.lfo_phase = 0.0

    def process(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_out = np.zeros_like(left)
        right_out = np.zeros_like(right)

        base_delay = (self.delay_ms / 1000) * self.sample_rate

        for i in range(len(left)):
            # LFO modulation
            lfo = np.sin(self.lfo_phase * 2 * np.pi)
            mod_delay = base_delay + lfo * self.depth * self.sample_rate
            mod_delay = np.clip(mod_delay, 1, self.buffer_size - 1)

            # Read from buffer with interpolation
            read_ptr = (self.write_ptr - mod_delay) % self.buffer_size
            read_ptr_int = int(np.floor(read_ptr))
            read_ptr_frac = read_ptr - read_ptr_int
            read_ptr_next = (read_ptr_int + 1) % self.buffer_size

            # Linear interpolation
            wet_l = (1 - read_ptr_frac) * self.buffer_l[read_ptr_int] + \
                    read_ptr_frac * self.buffer_l[read_ptr_next]
            wet_r = (1 - read_ptr_frac) * self.buffer_r[read_ptr_int] + \
                    read_ptr_frac * self.buffer_r[read_ptr_next]

            # Write to buffer with feedback
            self.buffer_l[self.write_ptr] = left[i] + wet_l * self.feedback
            self.buffer_r[self.write_ptr] = right[i] + wet_r * self.feedback

            # Mix output
            left_out[i] = (1 - self.mix) * left[i] + self.mix * wet_l
            right_out[i] = (1 - self.mix) * right[i] + self.mix * wet_r

            self.write_ptr = (self.write_ptr + 1) % self.buffer_size
            self.lfo_phase += self.rate / self.sample_rate
            self.lfo_phase %= 1.0

        return left_out, right_out


class Chorus:
    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.rate = 0.5
        self.depth = 0.002
        self.mix = 0.3
        self.delay_buffer = np.zeros(int(sample_rate * 0.05))
        self.write_ptr = 0
        self.lfo_phase = 0.0

    def process(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_out = np.zeros_like(left)
        right_out = np.zeros_like(right)

        for i in range(len(left)):
            # Write to buffer
            self.delay_buffer[self.write_ptr] = (left[i] + right[i]) / 2
            self.write_ptr = (self.write_ptr + 1) % len(self.delay_buffer)

            # Read with modulation
            mod = np.sin(self.lfo_phase * 2 * np.pi) * self.depth * self.sample_rate
            delay_samples = int(0.01 * self.sample_rate + mod)
            read_ptr = (self.write_ptr - delay_samples) % len(self.delay_buffer)

            wet = self.delay_buffer[read_ptr]
            left_out[i] = left[i] + wet * self.mix
            right_out[i] = right[i] + wet * self.mix

            self.lfo_phase += self.rate / self.sample_rate
            self.lfo_phase %= 1.0

        return left_out, right_out


class Delay:
    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.time = 0.25
        self.feedback = 0.3
        self.mix = 0.3
        self.buffer_size = int(sample_rate * 2)
        self.buffer_l = np.zeros(self.buffer_size)
        self.buffer_r = np.zeros(self.buffer_size)
        self.write_ptr = 0

    def process(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_out = np.zeros_like(left)
        right_out = np.zeros_like(right)
        delay_samples = int(self.time * self.sample_rate)

        for i in range(len(left)):
            read_ptr = (self.write_ptr - delay_samples) % self.buffer_size

            wet_l = self.buffer_l[read_ptr]
            wet_r = self.buffer_r[read_ptr]

            self.buffer_l[self.write_ptr] = left[i] + wet_l * self.feedback
            self.buffer_r[self.write_ptr] = right[i] + wet_r * self.feedback

            left_out[i] = left[i] * (1 - self.mix) + wet_l * self.mix
            right_out[i] = right[i] * (1 - self.mix) + wet_r * self.mix

            self.write_ptr = (self.write_ptr + 1) % self.buffer_size

        return left_out, right_out


class Reverb:
    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.room_size = 0.5
        self.damping = 0.5
        self.mix = 0.3
        self.comb_delays = [0.0297, 0.0371, 0.0411, 0.0437]
        self.allpass_delays = [0.005, 0.0017]
        self._init_buffers()

    def _init_buffers(self):
        self.comb_buffers = [np.zeros(int(d * self.sample_rate)) for d in self.comb_delays]
        self.allpass_buffers = [np.zeros(int(d * self.sample_rate)) for d in self.allpass_delays]
        self.comb_ptrs = [0] * len(self.comb_delays)
        self.allpass_ptrs = [0] * len(self.allpass_delays)

    def process(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mono = (left + right) / 2
        output = np.zeros_like(mono)

        for i in range(len(mono)):
            # Comb filters
            comb_sum = 0
            for j, (delay, buffer) in enumerate(zip(self.comb_delays, self.comb_buffers)):
                ptr = self.comb_ptrs[j]
                comb_sum += buffer[ptr]
                buffer[ptr] = mono[i] + buffer[ptr] * self.room_size * (1 - self.damping)
                self.comb_ptrs[j] = (ptr + 1) % len(buffer)

            # Allpass filters
            ap_out = comb_sum / len(self.comb_delays)
            for j, (delay, buffer) in enumerate(zip(self.allpass_delays, self.allpass_buffers)):
                ptr = self.allpass_ptrs[j]
                ap_in = ap_out
                ap_out = -ap_in + buffer[ptr]
                buffer[ptr] = ap_in + ap_out * 0.5
                self.allpass_ptrs[j] = (ptr + 1) % len(buffer)

            output[i] = ap_out

        left_out = left * (1 - self.mix) + output * self.mix
        right_out = right * (1 - self.mix) + output * self.mix

        return left_out, right_out


class Compressor:
    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.threshold = -20.0
        self.ratio = 4.0
        self.attack = 0.01
        self.release = 0.1
        self.makeup_gain = 0.0
        self.envelope = 0.0

    def process(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_out = np.zeros_like(left)
        right_out = np.zeros_like(right)

        attack_coef = np.exp(-1.0 / (self.attack * self.sample_rate))
        release_coef = np.exp(-1.0 / (self.release * self.sample_rate))

        threshold_linear = 10 ** (self.threshold / 20)

        for i in range(len(left)):
            # Detect level
            level = max(abs(left[i]), abs(right[i]))

            if level > self.envelope:
                self.envelope = attack_coef * self.envelope + (1 - attack_coef) * level
            else:
                self.envelope = release_coef * self.envelope + (1 - release_coef) * level

            # Calculate gain reduction
            if self.envelope > threshold_linear:
                gain_reduction = threshold_linear + (self.envelope - threshold_linear) / self.ratio
                gain = gain_reduction / self.envelope
            else:
                gain = 1.0

            makeup_gain_linear = 10 ** (self.makeup_gain / 20)
            left_out[i] = left[i] * gain * makeup_gain_linear
            right_out[i] = right[i] * gain * makeup_gain_linear

        return left_out, right_out


class EQ:
    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.bands = [
            {'freq': 100, 'gain': 0.0, 'q': 0.7, 'type': 'lowshelf'},
            {'freq': 1000, 'gain': 0.0, 'q': 1.0, 'type': 'peak'},
            {'freq': 5000, 'gain': 0.0, 'q': 1.0, 'type': 'peak'},
            {'freq': 10000, 'gain': 0.0, 'q': 0.7, 'type': 'highshelf'}
        ]
        self._init_filters()

    def _init_filters(self):
        self.filters_l = []
        self.filters_r = []
        for band in self.bands:
            if band['type'] == 'peak':
                b, a = signal.iirpeak(band['freq'], band['q'], self.sample_rate)
            elif band['type'] == 'lowshelf':
                b, a = signal.iirfilter(2, band['freq'] / (self.sample_rate/2), btype='low')
            else:
                b, a = signal.iirfilter(2, band['freq'] / (self.sample_rate/2), btype='high')

            self.filters_l.append({'b': b, 'a': a, 'z': signal.lfilter_zi(b, a)})
            self.filters_r.append({'b': b, 'a': a, 'z': signal.lfilter_zi(b, a)})

    def process(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_out = left.copy()
        right_out = right.copy()

        for i, band in enumerate(self.bands):
            if band['gain'] != 0:
                gain_linear = 10 ** (band['gain'] / 20)
                left_out, _ = signal.lfilter(self.filters_l[i]['b'], self.filters_l[i]['a'],
                                            left_out, zi=self.filters_l[i]['z'] * left_out[0])
                right_out, _ = signal.lfilter(self.filters_r[i]['b'], self.filters_r[i]['a'],
                                             right_out, zi=self.filters_r[i]['z'] * right_out[0])
                left_out *= gain_linear
                right_out *= gain_linear

        return left_out, right_out


class FXChain:
    def __init__(self, sample_rate: float = 44100):
        self.distortion = Distortion()
        self.phaser = Phaser(sample_rate)
        self.flanger = Flanger(sample_rate)
        self.chorus = Chorus(sample_rate)
        self.delay = Delay(sample_rate)
        self.reverb = Reverb(sample_rate)
        self.compressor = Compressor(sample_rate)
        self.eq = EQ(sample_rate)
        self.order = ['distortion', 'eq', 'phaser', 'flanger', 'chorus', 'delay', 'reverb', 'compressor']
        self.enabled = {fx: True for fx in self.order}

    def process(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out_l, out_r = left.copy(), right.copy()

        for fx_name in self.order:
            if self.enabled.get(fx_name, False):
                fx = getattr(self, fx_name)
                out_l, out_r = fx.process(out_l, out_r)

        return out_l, out_r


# ============================================================================
# VOICE AND RENDERER
# ============================================================================

class Voice:
    def __init__(self, sample_rate: float, note: int, velocity: float):
        self.sample_rate = sample_rate
        self.note = note
        self.velocity = velocity
        self.osc_bank = OscillatorBank(sample_rate)
        self.amp_env = Envelope(sample_rate)
        self.age = 0
        self.active = True

    def trigger(self):
        self.amp_env.trigger()

    def release(self):
        self.amp_env.release_note()

    def process(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.active:
            return np.zeros(num_samples), np.zeros(num_samples)

        # Generate oscillator audio
        freq = 440.0 * 2 ** ((self.note - 69) / 12.0)
        left, right = self.osc_bank.process_block(freq, num_samples)

        # Apply amplitude envelope
        amp_env = self.amp_env.process(num_samples)
        left *= amp_env * self.velocity
        right *= amp_env * self.velocity

        self.age += num_samples

        # Check if voice should be deactivated
        if self.amp_env.stage == 'idle' and self.age > self.sample_rate * 0.1:
            self.active = False

        return left, right


class Renderer:
    def __init__(self, sample_rate: float = 44100, polyphony: int = 8):
        self.sample_rate = sample_rate
        self.polyphony = polyphony
        self.voices: List[Voice] = []
        self.fx_chain = FXChain(sample_rate)
        self.master_volume = 0.8

        # Modulation
        self.lfo1 = LFO(sample_rate)
        self.lfo2 = LFO(sample_rate)
        self.mod_matrix = ModulationMatrix()

    def note_on(self, note: int, velocity: float = 1.0):
        # Remove oldest voice if at polyphony limit
        if len(self.voices) >= self.polyphony:
            oldest_voice = min(self.voices, key=lambda v: v.age)
            self.voices.remove(oldest_voice)

        voice = Voice(self.sample_rate, note, velocity)
        voice.trigger()
        self.voices.append(voice)

    def note_off(self, note: int):
        for voice in self.voices:
            if voice.note == note and voice.active:
                voice.release()

    def render_note_sequence(self, notes: List[Tuple[float, int, float, float]],
                            duration: float) -> np.ndarray:
        """
        Render a sequence of notes.
        notes: List of (start_time, note_number, duration, velocity)
        duration: Total render duration in seconds
        """
        total_samples = int(duration * self.sample_rate)
        left_out = np.zeros(total_samples)
        right_out = np.zeros(total_samples)

        # Schedule notes
        scheduled_notes = []
        for start, note, note_dur, vel in notes:
            start_sample = int(start * self.sample_rate)
            end_sample = int((start + note_dur) * self.sample_rate)
            scheduled_notes.append((start_sample, end_sample, note, vel))

        scheduled_notes.sort(key=lambda x: x[0])

        # Process block by block for efficiency
        block_size = 256
        current_notes = []

        for block_start in range(0, total_samples, block_size):
            block_end = min(block_start + block_size, total_samples)
            block_samples = block_end - block_start

            # Check for new notes
            while scheduled_notes and scheduled_notes[0][0] < block_end:
                start, end, note, vel = scheduled_notes.pop(0)
                current_notes.append((end, note, vel))
                self.note_on(note, vel)

            # Check for note offs
            current_notes = [(end, note, vel) for end, note, vel in current_notes if end > block_start]
            for end, note, vel in current_notes:
                if end < block_end:
                    self.note_off(note)

            # Process active voices
            block_left = np.zeros(block_samples)
            block_right = np.zeros(block_samples)

            for voice in self.voices[:]:
                if voice.active:
                    v_left, v_right = voice.process(block_samples)
                    block_left += v_left
                    block_right += v_right
                else:
                    self.voices.remove(voice)

            # Apply FX chain
            block_left, block_right = self.fx_chain.process(block_left, block_right)

            # Write to output
            left_out[block_start:block_end] = block_left * self.master_volume
            right_out[block_start:block_end] = block_right * self.master_volume

        return np.column_stack([left_out, right_out])

    def render(self, note_sequence: List[Tuple[int, float, float]],
               duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Simplified render interface.
        note_sequence: List of (note_number, start_time, duration)
        """
        notes_with_velocity = [(start, note, dur, 1.0) for note, start, dur in note_sequence]
        return self.render_note_sequence(notes_with_velocity, duration)

    def export_wav(self, audio: np.ndarray, filename: str):
        """Export audio to WAV file."""
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, audio_int16)


# ============================================================================
# PRESET SYSTEM
# ============================================================================

class PresetManager:
    def __init__(self, renderer: Renderer):
        self.renderer = renderer

    def save_preset(self, filename: str):
        """Save current synthesizer state to JSON."""
        preset = {
            'osc1': {
                'wave_type': int(self.renderer.voices[0].osc_bank.osc1_config.wave_type) if self.renderer.voices else 1,
                'level': self.renderer.voices[0].osc_bank.osc1_config.level if self.renderer.voices else 1.0,
                'coarse': self.renderer.voices[0].osc_bank.osc1_config.coarse_pitch if self.renderer.voices else 0.0,
                'fine': self.renderer.voices[0].osc_bank.osc1_config.fine_pitch if self.renderer.voices else 0.0,
                'unison': self.renderer.voices[0].osc_bank.osc1_config.unison_voices if self.renderer.voices else 1,
            },
            'fx': {
                'distortion_drive': self.renderer.fx_chain.distortion.drive,
                'delay_mix': self.renderer.fx_chain.delay.mix,
                'reverb_mix': self.renderer.fx_chain.reverb.mix,
                'phaser_mix': self.renderer.fx_chain.phaser.mix,
            }
        }

        with open(filename, 'w') as f:
            json.dump(preset, f, indent=2)

    def load_preset(self, filename: str):
        """Load synthesizer state from JSON."""
        with open(filename, 'r') as f:
            preset = json.load(f)

        # Apply to all voices
        for voice in self.renderer.voices:
            voice.osc_bank.osc1_config.wave_type = preset['osc1']['wave_type']
            voice.osc_bank.osc1_config.level = preset['osc1']['level']
            voice.osc_bank.osc1_config.coarse_pitch = preset['osc1']['coarse']
            voice.osc_bank.osc1_config.fine_pitch = preset['osc1']['fine']
            voice.osc_bank.osc1_config.unison_voices = preset['osc1']['unison']

        self.renderer.fx_chain.distortion.drive = preset['fx']['distortion_drive']
        self.renderer.fx_chain.delay.mix = preset['fx']['delay_mix']
        self.renderer.fx_chain.reverb.mix = preset['fx']['reverb_mix']
        if 'phaser_mix' in preset['fx']:
            self.renderer.fx_chain.phaser.mix = preset['fx']['phaser_mix']


# ============================================================================
# PATCH GENERATOR
# ============================================================================

class RandomPatchGenerator:
    def __init__(self, renderer: Renderer):
        self.renderer = renderer

    def generate_random_patch(self):
        """Generate random synthesizer patch."""
        if not self.renderer.voices:
            return

        voice = self.renderer.voices[0] if self.renderer.voices else None
        if voice is None:
            return

        # Random OSC1
        voice.osc_bank.osc1_config.wave_type = np.random.choice([1, 2, 3, 4, 5, 6])
        voice.osc_bank.osc1_config.level = np.random.uniform(0.5, 1.0)
        voice.osc_bank.osc1_config.coarse_pitch = np.random.choice([-12, 0, 12, 7, 19])
        voice.osc_bank.osc1_config.unison_voices = np.random.choice([1, 2, 3, 4, 5, 7])
        voice.osc_bank.osc1_config.unison_detune = np.random.uniform(0, 30)

        # Random OSC2
        voice.osc_bank.osc2_config.wave_type = np.random.choice([1, 2, 3, 4, 5, 6])
        voice.osc_bank.osc2_config.level = np.random.uniform(0.3, 0.8)
        voice.osc_bank.osc2_config.coarse_pitch = np.random.choice([-7, 0, 7, 12])

        # Random Envelopes
        voice.amp_env.attack = np.random.uniform(0.001, 0.1)
        voice.amp_env.decay = np.random.uniform(0.1, 0.5)
        voice.amp_env.sustain = np.random.uniform(0.3, 0.8)
        voice.amp_env.release = np.random.uniform(0.1, 1.0)

        # Random FX
        self.renderer.fx_chain.distortion.drive = np.random.uniform(0, 15)
        self.renderer.fx_chain.delay.mix = np.random.uniform(0, 0.4)
        self.renderer.fx_chain.delay.time = np.random.choice([0.125, 0.25, 0.375, 0.5])
        self.renderer.fx_chain.reverb.mix = np.random.uniform(0, 0.3)
        self.renderer.fx_chain.reverb.room_size = np.random.uniform(0.3, 0.8)
        self.renderer.fx_chain.phaser.mix = np.random.uniform(0, 0.6)


# ============================================================================
# MAIN SYNTHESIZER CLASS
# ============================================================================

class SerumBrother:
    """
    Main synthesizer interface - Serum Brother Wavetable Engine
    Professional-grade offline wavetable synthesis
    """

    def __init__(self, sample_rate: int = 44100, polyphony: int = 8):
        self.sample_rate = sample_rate
        self.renderer = Renderer(sample_rate, polyphony)
        self.preset_manager = PresetManager(self.renderer)
        self.patch_generator = RandomPatchGenerator(self.renderer)

        # Macro controls
        self.macros = [0.5, 0.5, 0.5, 0.5]

    def render(self, note_sequence: List[Tuple[int, float, float]],
               duration: float) -> np.ndarray:
        """
        Render audio from note sequence.

        Args:
            note_sequence: List of (midi_note, start_time, duration)
            duration: Total audio duration in seconds

        Returns:
            Stereo audio array of shape (samples, 2)
        """
        return self.renderer.render(note_sequence, duration, self.sample_rate)

    def export(self, audio: np.ndarray, filename: str):
        """Export audio to WAV file."""
        self.renderer.export_wav(audio, filename)

    def note_on(self, note: int, velocity: float = 1.0):
        """Trigger a note."""
        self.renderer.note_on(note, velocity)

    def note_off(self, note: int):
        """Release a note."""
        self.renderer.note_off(note)

    def save_preset(self, filename: str):
        """Save current patch to JSON file."""
        self.preset_manager.save_preset(filename)

    def load_preset(self, filename: str):
        """Load patch from JSON file."""
        self.preset_manager.load_preset(filename)

    def randomize_patch(self):
        """Generate random synthesizer patch."""
        self.patch_generator.generate_random_patch()

    def set_oscillator(self, osc_num: int, wave_type: int, level: float = 1.0,
                      coarse: float = 0.0, fine: float = 0.0, unison: int = 1):
        """Configure an oscillator."""
        if not self.renderer.voices:
            voice = Voice(self.sample_rate, 60, 1.0)
            self.renderer.voices.append(voice)

        config = (self.renderer.voices[0].osc_bank.osc1_config if osc_num == 1
                  else self.renderer.voices[0].osc_bank.osc2_config)
        config.wave_type = wave_type
        config.level = level
        config.coarse_pitch = coarse
        config.fine_pitch = fine
        config.unison_voices = unison

    def set_envelope(self, attack: float = 0.01, decay: float = 0.1,
                    sustain: float = 0.7, release: float = 0.5):
        """Configure amplitude envelope."""
        for voice in self.renderer.voices:
            voice.amp_env.attack = attack
            voice.amp_env.decay = decay
            voice.amp_env.sustain = sustain
            voice.amp_env.release = release

    def set_fx(self, distortion: float = 0.0, delay_mix: float = 0.0,
               reverb_mix: float = 0.0, phaser_mix: float = 0.0):
        """Configure effects chain."""
        self.renderer.fx_chain.distortion.drive = distortion
        self.renderer.fx_chain.delay.mix = delay_mix
        self.renderer.fx_chain.reverb.mix = reverb_mix
        self.renderer.fx_chain.phaser.mix = phaser_mix


# ============================================================================
# EXAMPLE USAGE - DISTORTED SUB WITH PHASER
# ============================================================================

if __name__ == "__main__":
    # Create synthesizer instance
    synth = SerumBrother(sample_rate=44100, polyphony=8)

    # ============================================
    # DISTORTED SUB BASS WITH PHASER
    # ============================================

    # OSC1: Pure sine for sub foundation
    synth.set_oscillator(1, WaveType.SINE, level=1.0, coarse=-12, unison=1)

    # OSC2: Triangle for harmonics
    synth.set_oscillator(2, WaveType.TRIANGLE, level=0.6, coarse=-12, fine=5, unison=1)

    # Enable sub oscillator
    if synth.renderer.voices:
        synth.renderer.voices[0].osc_bank.sub_config.enabled = True
        synth.renderer.voices[0].osc_bank.sub_config.level = 0.7

    # Punchy envelope
    synth.set_envelope(attack=0.005, decay=0.3, sustain=0.7, release=0.4)

    # Heavy distortion + phaser
    synth.set_fx(distortion=12.0, delay_mix=0.15, reverb_mix=0.1, phaser_mix=0.7)

    # Configure phaser specifically for sub
    synth.renderer.fx_chain.phaser.rate = 0.3
    synth.renderer.fx_chain.phaser.depth = 0.8
    synth.renderer.fx_chain.phaser.feedback = 0.5
    synth.renderer.fx_chain.phaser.f_min = 100
    synth.renderer.fx_chain.phaser.f_max = 800

    # Disable other FX for cleaner sound
    synth.renderer.fx_chain.enabled['flanger'] = False
    synth.renderer.fx_chain.enabled['chorus'] = False
    synth.renderer.fx_chain.enabled['eq'] = False
    synth.renderer.fx_chain.enabled['compressor'] = False

    # Create sub bassline
    note_sequence = [
        (36, 0.0, 0.5),   # C1
        (36, 0.5, 0.25),  # C1
        (39, 0.75, 0.25), # D#2
        (41, 1.0, 0.5),   # F2
        (36, 1.5, 0.5),   # C1
        (43, 2.0, 0.5),   # G2
        (41, 2.5, 0.5),   # F2
        (39, 3.0, 0.5),   # D#2
    ]

    # Render audio
    print("Rendering distorted sub bass with phaser...")
    audio = synth.render(note_sequence, duration=4.0)

    # Export to WAV
    synth.export(audio, "distorted_sub_phaser.wav")
    print("✅ Exported to distorted_sub_phaser.wav")

    # Save preset
    synth.save_preset("distorted_sub_phaser.json")
    print("✅ Preset saved to distorted_sub_phaser.json")