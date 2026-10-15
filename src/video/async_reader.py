"""Bounded threaded OpenCV video reader."""
import queue, threading
from dataclasses import dataclass
import cv2

@dataclass(frozen=True)
class FramePacket:
    index:int
    timestamp_seconds:float
    frame:object

class AsyncVideoReader:
    _END=object()
    def __init__(self, source, queue_size=16, drop_oldest=False, loop_video=False):
        if queue_size<1:raise ValueError('queue_size must be positive')
        self.source=source; self.frames=queue.Queue(queue_size); self.drop_oldest=drop_oldest
        self.loop_video=loop_video; self._stop=threading.Event(); self._thread=None; self._capture=None; self.error=None
    def start(self):
        if self._thread and self._thread.is_alive():return self
        self._stop.clear(); self.error=None
        while not self.frames.empty():
            try:self.frames.get_nowait()
            except queue.Empty:break
        self._thread=threading.Thread(target=self._decode,name='raven-video-reader',daemon=True); self._thread.start(); return self
    def _put(self,item):
        while not self._stop.is_set():
            try:self.frames.put(item,timeout=.05); return True
            except queue.Full:
                if self.drop_oldest:
                    try:self.frames.get_nowait()
                    except queue.Empty:pass
        return False
    def _decode(self):
        try:
            self._capture=cv2.VideoCapture(self.source)
            if not self._capture.isOpened():raise RuntimeError(f'could not open video source: {self.source}')
            fps=float(self._capture.get(cv2.CAP_PROP_FPS) or 0) or 30.; i=0
            while not self._stop.is_set():
                ok,frame=self._capture.read()
                if not ok:
                    if self.loop_video and self._capture.set(cv2.CAP_PROP_POS_FRAMES,0):continue
                    break
                ms=float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0)
                if not self._put(FramePacket(i,ms/1000 if ms>0 else i/fps,frame)):break
                i+=1
        except Exception as e:self.error=e
        finally:
            if self._capture is not None:self._capture.release()
            while not self._stop.is_set():
                try:self.frames.put(self._END,timeout=.05);break
                except queue.Full:
                    if self.drop_oldest:
                        try:self.frames.get_nowait()
                        except queue.Empty:pass
    def read(self,timeout=None):
        if self._thread is None:self.start()
        try:item=self.frames.get(timeout=timeout)
        except queue.Empty:
            if self._thread and not self._thread.is_alive() and self.frames.empty():return None
            raise TimeoutError('timed out waiting for decoded frame')
        if item is self._END:
            if self.error:
                raise RuntimeError(f"video decode failed: {self.error}") from self.error
            return None
        return item
    def close(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():self._thread.join(timeout=2)
        # The decoding thread owns capture lifetime; release here only as a
        # last resort if a backend is stalled during shutdown.
        if self._thread and self._thread.is_alive() and self._capture is not None:
            self._capture.release()
            self._thread.join(timeout=1)
        while not self.frames.empty():
            try:self.frames.get_nowait()
            except queue.Empty:break
    def __enter__(self):return self.start()
    def __exit__(self,*args):self.close()
