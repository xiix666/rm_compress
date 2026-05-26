"""Python reader for rmcompress::ShmRing shared memory.

Matches onboard/librmcompress/src/shm_ring.cpp exactly.
The C++ camera_capture writes BGR frames; this reader reads them.
"""
import mmap, os, struct

SHM_RING_MAGIC = 0x524d5348  # "RMSH"
SHM_RING_VERSION = 1
SHM_PIXFMT_BGR8 = 0x42475238

# struct ShmRingHeader: 56 bytes
_HDR_FMT = "<IIII IIII QQQ"
_HDR_SIZE = struct.calcsize(_HDR_FMT)  # 56

# struct ShmFrameHeader: 40 bytes
_FRM_FMT = "<QQ IIII II"
_FRM_SIZE = struct.calcsize(_FRM_FMT)  # 40


class ShmFrame:
    __slots__ = ("data", "sequence", "timestamp_ns", "width", "height",
                 "stride", "pixfmt", "data_bytes")
    def __init__(self):
        self.data = None
        self.sequence = 0
        self.timestamp_ns = 0
        self.width = 0
        self.height = 0
        self.stride = 0
        self.pixfmt = 0
        self.data_bytes = 0


class ShmRingReader:
    def __init__(self, name="/rm_camera_frames"):
        if not name.startswith("/"):
            name = "/" + name
        # Open shm via /dev/shm/<name> (POSIX shm_open equivalent)
        shm_path = "/dev/shm" + name
        fd = os.open(shm_path, os.O_RDWR)
        try:
            st = os.fstat(fd)
            self._mapping_size = st.st_size
            if self._mapping_size < _HDR_SIZE:
                raise ValueError(f"shm too small: {self._mapping_size}B")
            self._mm = mmap.mmap(fd, self._mapping_size,
                                 mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)

        # Parse header
        hdr = self._mm[:_HDR_SIZE]
        vals = struct.unpack(_HDR_FMT, hdr)
        if vals[0] != SHM_RING_MAGIC or vals[1] != SHM_RING_VERSION:
            raise ValueError(f"shm magic/version mismatch: {vals[0]:08x}/{vals[1]}")
        self.capacity = vals[2]
        self.slot_bytes = vals[3]
        self.width = vals[4]
        self.height = vals[5]
        self.stride = vals[6]
        self.pixfmt = vals[7]
        self._last_sequence = 0

    def read_latest(self):
        """Return ShmFrame or None if no new frame available.

        Matches C++ ShmRing::read_latest() double-check sequence protocol.
        """
        hdr = self._mm[:_HDR_SIZE]
        vals = struct.unpack(_HDR_FMT, hdr)
        seq = vals[8]  # write_sequence
        if seq == 0 or seq == self._last_sequence:
            return None

        index = seq % self.capacity
        slot_offset = _HDR_SIZE + index * self.slot_bytes
        slot = self._mm[slot_offset:slot_offset + self.slot_bytes]

        # Read frame header
        frm_vals = struct.unpack(_FRM_FMT, slot[:_FRM_SIZE])
        a = frm_vals[0]  # sequence
        if (a & 1) != 0 or (a >> 1) != seq:
            return None  # writer in progress or stale

        data_bytes = frm_vals[6]  # data_bytes
        data = slot[_FRM_SIZE:_FRM_SIZE + data_bytes]

        # Double-check: re-read sequence to ensure no concurrent write
        slot2 = self._mm[slot_offset:slot_offset + _FRM_SIZE]
        frm_vals2 = struct.unpack(_FRM_FMT, slot2)
        b = frm_vals2[0]
        if a != b or (b & 1) != 0 or (b >> 1) != seq:
            return None  # torn write

        self._last_sequence = seq
        sf = ShmFrame()
        sf.sequence = seq
        sf.timestamp_ns = frm_vals[1]
        sf.width = frm_vals[2]
        sf.height = frm_vals[3]
        sf.stride = frm_vals[4]
        sf.pixfmt = frm_vals[5]
        sf.data_bytes = frm_vals[6]
        sf.data = data
        return sf

    def close(self):
        if self._mm:
            self._mm.close()
            self._mm = None
