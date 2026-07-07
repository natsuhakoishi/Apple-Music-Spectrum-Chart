import sys
import asyncio
import threading
import numpy as np
import pyaudiowpatch as pyaudio
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import (QPainter, QColor, QPen, QLinearGradient, QFont, QPixmap, QPolygon)
from PyQt5.QtCore import QTimer, Qt, QRect, QPoint

CAPTURE_CHUNK    = 512
FFT_SIZE         = 16384
NUM_BARS         = 64
DB_FLOOR         = -70.0
DB_CEIL          = -5.0
PEAK_HOLD_FRAMES = 30
PEAK_DROP_SPEED  = 0.015
PLATE_H          = 3
PROGRESS_H       = 4
PROGRESS_PAD     = 30


def _draw_prev(p, cx, cy, size, color):
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    p.drawRect(int(cx - size), int(cy - size//2), max(2, size//4), size)
    p.drawPolygon(QPolygon([
        QPoint(int(cx - size + size//4), cy),
        QPoint(int(cx), int(cy - size//2)),
        QPoint(int(cx), int(cy + size//2)),
    ]))
    p.drawPolygon(QPolygon([
        QPoint(int(cx), cy),
        QPoint(int(cx + size), int(cy - size//2)),
        QPoint(int(cx + size), int(cy + size//2)),
    ]))


def _draw_next(p, cx, cy, size, color):
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    p.drawPolygon(QPolygon([
        QPoint(int(cx - size), int(cy - size//2)),
        QPoint(int(cx - size), int(cy + size//2)),
        QPoint(int(cx), cy),
    ]))
    p.drawPolygon(QPolygon([
        QPoint(int(cx), int(cy - size//2)),
        QPoint(int(cx), int(cy + size//2)),
        QPoint(int(cx + size), cy),
    ]))
    p.drawRect(int(cx + size), int(cy - size//2), max(2, size//4), size)


def _draw_pause(p, cx, cy, size, color):
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    bar_w = max(3, size // 3)
    gap   = max(2, size // 4)
    p.drawRoundedRect(int(cx - gap//2 - bar_w), int(cy - size//2), bar_w, size, 1, 1)
    p.drawRoundedRect(int(cx + gap//2),          int(cy - size//2), bar_w, size, 1, 1)


def _draw_play(p, cx, cy, size, color):
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    p.drawPolygon(QPolygon([
        QPoint(int(cx - size//2), int(cy - size//2)),
        QPoint(int(cx - size//2), int(cy + size//2)),
        QPoint(int(cx + size//2), cy),
    ]))


class SpectrumWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setWindowTitle("Apple Music Spectrum")
        self.resize(520, 320)

        self._btn_apple = QRect()

        self._resize_edge  = None
        self._resize_start = None
        self._resize_geom  = None
        RESIZE_MARGIN      = 7

        self._btn_close    = QRect()
        self._btn_minimize = QRect()

        self.levels      = np.zeros(NUM_BARS)
        self.global_peak = DB_FLOOR
        self.peak_levels = np.zeros(NUM_BARS)
        self.peak_timers = np.zeros(NUM_BARS, dtype=int)

        self._ring        = np.zeros(FFT_SIZE, dtype=np.float32)
        self._audio_lock  = threading.Lock()
        self._result_lock = threading.Lock()
        self.normalized   = np.zeros(NUM_BARS)
        self._new_data    = threading.Event()

        self.now_title     = ""
        self.now_artist    = ""
        self.now_thumbnail = None
        self.is_playing    = True

        self.progress_pos      = 0.0
        self.progress_duration = 0.0

        self._current_session = None
        self._session_lock    = threading.Lock()

        self._ctrl_loop = asyncio.new_event_loop()
        threading.Thread(target=self._ctrl_loop.run_forever, daemon=True).start()

        self.pa          = pyaudio.PyAudio()
        wasapi_info      = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = self.pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if not default_speakers.get("isLoopbackDevice", False):
            for loopback in self.pa.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break

        self.rate     = int(default_speakers["defaultSampleRate"])
        self.channels = default_speakers["maxInputChannels"]

        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=default_speakers["index"],
            frames_per_buffer=CAPTURE_CHUNK,
            stream_callback=self.audio_callback,
        )
        self.stream.start_stream()

        self._running = True
        threading.Thread(target=self._fft_worker,   daemon=True).start()
        threading.Thread(target=self._media_worker, daemon=True).start()

        self._paint_timer = QTimer()
        self._paint_timer.timeout.connect(self._smooth_and_repaint)
        self._paint_timer.setTimerType(0)
        self._paint_timer.start(17)

        self._btn_prev  = QRect()
        self._btn_play  = QRect()
        self._btn_next  = QRect()
        self._btn_hover = None
        self.setMouseTracking(True)

    def _get_edge(self, pos):
        x, y, w, h = pos.x(), pos.y(), self.width(), self.height()
        m = 6
        left   = x < m
        right  = x > w - m
        top    = y < m
        bottom = y > h - m
        if top    and left:  return 'tl'
        if top    and right: return 'tr'
        if bottom and left:  return 'bl'
        if bottom and right: return 'br'
        if left:             return 'l'
        if right:            return 'r'
        if top:              return 't'
        if bottom:           return 'b'
        return None

    def _send_command(self, coro_fn):
        async def _run():
            with self._session_lock:
                session = self._current_session
            if session:
                try:
                    await coro_fn(session)
                except Exception:
                    pass
        asyncio.run_coroutine_threadsafe(_run(), self._ctrl_loop)

    def cmd_prev(self):
        self._send_command(lambda s: s.try_skip_previous_async())

    def cmd_next(self):
        self._send_command(lambda s: s.try_skip_next_async())

    def cmd_play_pause(self):
        if self.is_playing:
            self._send_command(lambda s: s.try_pause_async())
        else:
            self._send_command(lambda s: s.try_play_async())
        self.is_playing = not self.is_playing
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            if self._btn_apple.contains(pos):
                import subprocess
                subprocess.Popen(['explorer.exe', 'itms://'])
                return
            elif self._btn_close.contains(pos):
                self.close()
                return
            elif self._btn_minimize.contains(pos):
                self.showMinimized()
                return
            elif self._btn_prev.contains(pos):
                self.cmd_prev()
                return
            elif self._btn_play.contains(pos):
                self.cmd_play_pause()
                return
            elif self._btn_next.contains(pos):
                self.cmd_next()
                return

            edge = self._get_edge(pos)
            if edge:
                self._resize_edge  = edge
                self._resize_start = event.globalPos()
                self._resize_geom  = self.geometry()
            else:
                self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        pos = event.pos()

        if event.buttons() == Qt.LeftButton and self._resize_edge:
            delta = event.globalPos() - self._resize_start
            g     = self._resize_geom
            x, y, w, h = g.x(), g.y(), g.width(), g.height()
            min_w, min_h = 300, 200
            e = self._resize_edge
            if 'r' in e: w = max(min_w, g.width()  + delta.x())
            if 'b' in e: h = max(min_h, g.height() + delta.y())
            if 'l' in e:
                new_w = max(min_w, g.width() - delta.x())
                x = g.x() + g.width() - new_w
                w = new_w
            if 't' in e:
                new_h = max(min_h, g.height() - delta.y())
                y = g.y() + g.height() - new_h
                h = new_h
            self.setGeometry(x, y, w, h)
            return

        if event.buttons() == Qt.LeftButton and hasattr(self, '_drag_pos'):
            self.move(event.globalPos() - self._drag_pos)
            return

        prev = self._btn_hover
        if self._btn_apple.contains(pos):
            self._btn_hover = "apple"
        elif self._btn_close.contains(pos):
            self._btn_hover = "close"
        elif self._btn_minimize.contains(pos):
            self._btn_hover = "minimize"
        elif self._btn_prev.contains(pos):
            self._btn_hover = "prev"
        elif self._btn_play.contains(pos):
            self._btn_hover = "play"
        elif self._btn_next.contains(pos):
            self._btn_hover = "next"
        else:
            self._btn_hover = None
        if self._btn_hover != prev:
            self.update()

        edge = self._get_edge(pos)
        cursors = {
            'tl': Qt.SizeFDiagCursor, 'br': Qt.SizeFDiagCursor,
            'tr': Qt.SizeBDiagCursor, 'bl': Qt.SizeBDiagCursor,
            'l':  Qt.SizeHorCursor,   'r':  Qt.SizeHorCursor,
            't':  Qt.SizeVerCursor,   'b':  Qt.SizeVerCursor,
        }
        self.setCursor(cursors.get(edge, Qt.ArrowCursor))

    def mouseReleaseEvent(self, event):
        self._resize_edge  = None
        self._resize_start = None
        self._resize_geom  = None
        if hasattr(self, '_drag_pos'):
            del self._drag_pos

    def leaveEvent(self, event):
        self._btn_hover = None
        self.update()

    def _media_worker(self):
        async def _poll():
            try:
                from winrt.windows.media.control import \
                    GlobalSystemMediaTransportControlsSessionManager as Manager
                import winrt.windows.storage.streams as streams

                while self._running:
                    title, artist, thumb, found_session = "", "", None, None
                    pos, duration = 0.0, 0.0
                    try:
                        mgr      = await Manager.request_async()
                        sessions = mgr.get_sessions()
                        for session in sessions:
                            try:
                                app_id = session.source_app_user_model_id or ""
                                info   = await session.try_get_media_properties_async()
                                if info and info.title:
                                    title         = info.title or ""
                                    raw_artist    = info.artist or ""
                                    artist        = raw_artist.split(" — ")[0].strip()
                                    found_session = session

                                    try:
                                        tl       = session.get_timeline_properties()
                                        start    = tl.start_time.total_seconds()
                                        end      = tl.end_time.total_seconds()
                                        cur      = tl.position.total_seconds()
                                        duration = end - start
                                        pos = max(0.0, min(1.0, (cur - start) / duration)) \
                                              if duration > 0 else 0.0
                                    except Exception:
                                        pos, duration = 0.0, 0.0

                                    try:
                                        thumb_ref = info.thumbnail
                                        if thumb_ref:
                                            stream = await thumb_ref.open_read_async()
                                            size   = stream.size
                                            reader = streams.DataReader(stream)
                                            await reader.load_async(size)
                                            buf = bytearray(size)
                                            reader.read_bytes(buf)
                                            pm = QPixmap()
                                            pm.loadFromData(bytes(buf))
                                            if not pm.isNull():
                                                thumb = pm
                                    except Exception:
                                        thumb = None

                                    if "apple" in app_id.lower():
                                        break
                            except Exception:
                                continue
                    except Exception:
                        pass

                    with self._session_lock:
                        self._current_session = found_session
                    with self._result_lock:
                        self.now_title         = title
                        self.now_artist        = artist
                        self.now_thumbnail     = thumb
                        self.progress_pos      = pos
                        self.progress_duration = duration

                    await asyncio.sleep(1)
            except ImportError:
                pass

        asyncio.run(_poll())

    def audio_callback(self, in_data, frame_count, time_info, status):
        arr = np.frombuffer(in_data, dtype=np.float32).copy()
        if self.channels > 1:
            arr = arr.reshape(-1, self.channels)
            mono = arr[:, 0].copy()
            for ch in range(1, self.channels):
                mono += arr[:, ch]
            arr = mono / self.channels
        n = len(arr)
        with self._audio_lock:
            self._ring[:-n] = self._ring[n:]
            self._ring[-n:] = arr[:n] if n <= FFT_SIZE else arr[-FFT_SIZE:]
        self._new_data.set()
        return (None, pyaudio.paContinue)

    def _fft_worker(self):
        log_min = np.log10(20.0)
        log_max = np.log10(min(self.rate / 2.0, 24000.0))
        edges   = np.logspace(log_min, log_max, NUM_BARS + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        window  = np.hanning(FFT_SIZE)
        freqs   = np.fft.rfftfreq(FFT_SIZE, 1.0 / self.rate)

        while self._running:
            self._new_data.wait(timeout=0.05)
            self._new_data.clear()
            with self._audio_lock:
                buf = self._ring.copy()

            fft    = np.abs(np.fft.rfft(buf * window)) / (FFT_SIZE / 2)
            fft_db = 20.0 * np.log10(fft + 1e-9)

            raw = np.full(NUM_BARS, DB_FLOOR)
            for i in range(NUM_BARS):
                mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
                if mask.any():
                    band_vals = fft_db[mask]
                    top_n  = min(3, len(band_vals))
                    raw[i] = float(np.mean(np.partition(band_vals, -top_n)[-top_n:]))
                else:
                    idx    = int(np.clip(np.searchsorted(freqs, centers[i]), 1, len(freqs) - 1))
                    f0, f1 = freqs[idx - 1], freqs[idx]
                    t      = (centers[i] - f0) / (f1 - f0) if f1 > f0 else 0.0
                    raw[i] = fft_db[idx - 1] + t * (fft_db[idx] - fft_db[idx - 1])

            frame_peak       = float(np.max(raw))
            self.global_peak = max(frame_peak, self.global_peak * 0.994)
            g_ceil           = max(self.global_peak, DB_FLOOR + 20)
            g_floor          = g_ceil - 60.0

            norm  = np.clip((raw - g_floor) / (g_ceil - g_floor), 0.0, 1.0)
            norm *= (raw > g_floor + 4)
            with self._result_lock:
                self.normalized = norm

    def _smooth_and_repaint(self):
        with self._result_lock:
            normalized = self.normalized.copy()
            title = self.now_title

        if title:
            self.setWindowTitle(title)

        for i in range(NUM_BARS):
            alpha = 0.75 if normalized[i] > self.levels[i] else 0.08
            self.levels[i] += (normalized[i] - self.levels[i]) * alpha
            if self.levels[i] >= self.peak_levels[i]:
                self.peak_levels[i] = self.levels[i]
                self.peak_timers[i] = PEAK_HOLD_FRAMES
            else:
                if self.peak_timers[i] > 0:
                    self.peak_timers[i] -= 1
                else:
                    self.peak_levels[i] = max(0.0, self.peak_levels[i] - PEAK_DROP_SPEED)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(22, 26, 40))

        W, H = self.width(), self.height()

        HEADER_H    = 32
        FOOTER_H    = 36
        BTN_W       = 48
        BTN_H       = FOOTER_H
        PROG_AREA_H = PROGRESS_H + 25

        p.fillRect(0, 0, W, HEADER_H, QColor(15, 18, 30))

        with self._result_lock:
            title    = self.now_title
            artist   = self.now_artist
            thumb    = self.now_thumbnail
            prog_pos = self.progress_pos

        font_title = QFont()
        font_title.setPointSize(9)
        font_title.setBold(True)
        p.setFont(font_title)
        p.setPen(QColor(240, 240, 255, 220))

        left_text  = title  if title else "Nothing playing"
        right_text = artist if title else ""
        left_w = int(W * 0.55)
        p.setClipRect(10, 0, left_w - 10, HEADER_H)
        p.drawText(10, 0, left_w - 10, HEADER_H, Qt.AlignVCenter | Qt.AlignLeft, left_text)
        p.setClipping(False)
        p.drawText(0, 0, W - 10, HEADER_H, Qt.AlignVCenter | Qt.AlignRight, right_text)

        p.setPen(QPen(QColor(255, 255, 255, 25), 1))
        p.drawLine(0, HEADER_H, W, HEADER_H)

        pad_l, pad_r = 30, 30
        pad_top      = HEADER_H + 16
        pad_bot      = FOOTER_H + PROG_AREA_H + 10
        chart_w = W - pad_l - pad_r
        chart_h = H - pad_top - pad_bot

        p.setPen(QPen(QColor(255, 255, 255, 18), 1))
        for i in range(1, 4):
            y = pad_top + chart_h * i // 4
            p.drawLine(pad_l, y, W - pad_r, y)

        bar_gap = 2
        bar_w   = (chart_w - bar_gap * (NUM_BARS - 1)) / NUM_BARS

        for i in range(NUM_BARS):
            x  = pad_l + i * (bar_w + bar_gap)
            bw = max(int(bar_w), 1)
            h  = self.levels[i] * chart_h
            y  = pad_top + chart_h - h

            grad = QLinearGradient(x, y, x, pad_top + chart_h)
            grad.setColorAt(0.0, QColor(255, 255, 255, 180))
            grad.setColorAt(1.0, QColor(255, 255, 255, 40))
            p.setBrush(grad)
            p.setPen(QPen(QColor(255, 255, 255, 50), 0.5))
            p.drawRoundedRect(int(x), int(y), bw, int(h), 2, 2)

            if self.peak_levels[i] > 0.01:
                py = int(pad_top + chart_h - self.peak_levels[i] * chart_h) - PLATE_H - 1
                p.setBrush(QColor(255, 255, 255, 200))
                p.setPen(QPen(QColor(255, 255, 255, 120), 0.5))
                p.drawRoundedRect(int(x), py, bw, PLATE_H, 1, 1)

        if thumb:
            chart_area_h = H - HEADER_H - FOOTER_H
            thumb_size   = int(min(chart_area_h * 0.6, W * 0.25, 200))
            scaled_thumb = thumb.scaled(
                thumb_size, thumb_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            margin = 8
            tx = W - scaled_thumb.width() - margin
            ty = HEADER_H + margin
            p.setOpacity(0.35)
            p.fillRect(tx - 2, ty - 2, scaled_thumb.width() + 4, scaled_thumb.height() + 4, QColor(0, 0, 0))
            p.setOpacity(1.0)
            p.drawPixmap(tx, ty, scaled_thumb)

        footer_y   = H - FOOTER_H
        prog_y     = footer_y - PROG_AREA_H + (PROG_AREA_H - PROGRESS_H) // 2
        bar_x      = PROGRESS_PAD
        bar_w_full = W - PROGRESS_PAD * 2

        p.setBrush(QColor(255, 255, 255, 30))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(bar_x, prog_y, bar_w_full, PROGRESS_H, PROGRESS_H // 2, PROGRESS_H // 2)

        filled_w = int(bar_w_full * prog_pos)
        if filled_w > 0:
            grad_prog = QLinearGradient(bar_x, 0, bar_x + bar_w_full, 0)
            grad_prog.setColorAt(0.0, QColor(180, 180, 255, 220))
            grad_prog.setColorAt(1.0, QColor(255, 255, 255, 200))
            p.setBrush(grad_prog)
            p.drawRoundedRect(bar_x, prog_y, filled_w, PROGRESS_H,
                              PROGRESS_H // 2, PROGRESS_H // 2)

        p.fillRect(0, footer_y, W, FOOTER_H, QColor(15, 18, 30))
        p.setPen(QPen(QColor(255, 255, 255, 25), 1))
        p.drawLine(0, footer_y, W, footer_y)

        cx = W // 2
        self._btn_play = QRect(cx - BTN_W // 2,              footer_y, BTN_W, BTN_H)
        self._btn_prev = QRect(cx - BTN_W // 2 - BTN_W - 8,  footer_y, BTN_W, BTN_H)
        self._btn_next = QRect(cx + BTN_W // 2 + 8,          footer_y, BTN_W, BTN_H)

        icon_size = 10
        for btn, draw_fn, key in [
            (self._btn_prev, lambda px, py: _draw_prev(p, px, py, icon_size, color), "prev"),
            (self._btn_play, lambda px, py: (
                _draw_pause(p, px, py, icon_size, color) if self.is_playing
                else _draw_play(p, px, py, icon_size, color)
            ), "play"),
            (self._btn_next, lambda px, py: _draw_next(p, px, py, icon_size, color), "next"),
        ]:
            color = QColor(255, 255, 255, 255) if self._btn_hover == key \
                    else QColor(255, 255, 255, 160)
            if self._btn_hover == key:
                p.setBrush(QColor(255, 255, 255, 25))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(btn, 6, 6)
            draw_fn(btn.x() + btn.width() // 2, btn.y() + btn.height() // 2)

        BTN_SZ = 12
        btn_y  = footer_y + (FOOTER_H - BTN_SZ) // 2
        self._btn_close    = QRect(W - BTN_SZ - 10,     btn_y, BTN_SZ, BTN_SZ)
        self._btn_minimize = QRect(W - BTN_SZ * 2 - 18, btn_y, BTN_SZ, BTN_SZ)

        for rect, hover_key, base_color in [
            (self._btn_close,    "close",    QColor(255, 80,  80)),
            (self._btn_minimize, "minimize", QColor(255, 180, 0)),
        ]:
            color = base_color if self._btn_hover == hover_key else QColor(80, 80, 100)
            p.setBrush(color)
            p.setPen(Qt.NoPen)
            p.drawEllipse(rect)

        APPLE_SZ = 24
        apple_x  = 8
        apple_y  = footer_y + (FOOTER_H - APPLE_SZ) // 2
        self._btn_apple = QRect(apple_x, apple_y, APPLE_SZ + 4, APPLE_SZ)

        apple_color = QColor(255, 255, 255, 255) if self._btn_hover == "apple" \
                      else QColor(255, 255, 255, 120)
        p.setPen(apple_color)
        font_apple = QFont()
        font_apple.setPointSize(20)
        p.drawText(self._btn_apple, Qt.AlignCenter, "♫")

        p.end()

    def closeEvent(self, event):
        self._running = False
        self._new_data.set()
        self._paint_timer.stop()
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        self._ctrl_loop.call_soon_threadsafe(self._ctrl_loop.stop)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SpectrumWidget()
    w.show()
    sys.exit(app.exec_())
