#!/usr/bin/env python3
"""Stream one /v1/audio/speech request, save wav, report TTFA and duration.
usage: smoke.py PORT OUT.wav [voice] [extra-json]"""
import json, sys, time, wave, struct, math, http.client
port = int(sys.argv[1]); out = sys.argv[2]
voice = sys.argv[3] if len(sys.argv) > 3 else 'Ryan'
body = {'model': 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
        'input': 'The quick brown fox jumps over the lazy dog.',
        'voice': voice, 'language': 'English',
        'stream': True, 'stream_format': 'audio', 'response_format': 'pcm'}
if len(sys.argv) > 4: body.update(json.loads(sys.argv[4]))
c = http.client.HTTPConnection('127.0.0.1', port, timeout=300)
t0 = time.perf_counter()
c.request('POST', '/v1/audio/speech', json.dumps(body), {'Content-Type': 'application/json'})
r = c.getresponse()
hdr = dict(r.getheaders())
if r.status != 200:
    print('HTTP', r.status, r.read()[:2000]); sys.exit(1)
buf = bytearray(); ttfb = None
while True:
    ch = r.read1(65536) if hasattr(r, 'read1') else r.read(4096)
    if not ch: break
    if ttfb is None: ttfb = time.perf_counter() - t0
    buf += ch
total = time.perf_counter() - t0
sr = 24000
if buf[:4] == b'RIFF':  # server returned wav despite pcm
    open(out, 'wb').write(buf); data = bytes(buf[44:])
else:
    data = bytes(buf[:len(buf)//2*2])
    with wave.open(out, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(data)
n = len(data)//2
s = struct.unpack('<%dh' % n, data) if n else ()
rms = math.sqrt(sum(x*x for x in s)/n) if n else 0.0
peak = max((abs(x) for x in s), default=0)
print(json.dumps({'ttfb_s': round(ttfb or -1, 4), 'total_s': round(total, 4), 'audio_s': round(n/sr, 3),
                  'rms': round(rms, 1), 'peak': peak, 'bytes': len(buf),
                  'content_type': hdr.get('content-type') or hdr.get('Content-Type')}))
