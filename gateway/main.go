// uqt-gw: HTTP gateway for the uqt GPU engine process.
//
// POST /v1/audio/speech {"input", "voice", "language"} -> chunked raw PCM (s16le, 24 kHz, mono).
// GET  /health -> 200 once the engine has reported READY.
//
// Requests and audio are multiplexed over one Unix-domain socket to the engine process (see
// uqt/engine_proc.py for the frame format). Each HTTP stream is a goroutine; the engine never touches
// HTTP, so the GPU loop does not compete with request handling for the Python interpreter lock.
package main

import (
	"bufio"
	"sort"
	"encoding/binary"
	"encoding/json"
	"flag"
	"io"
	"log"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

const (
	tReq    = 1
	tAudio  = 2
	tEnd    = 3
	tErr    = 4
	tCancel = 5
	tReady  = 6
)

type hopStats struct {
	mu                    sync.Mutex
	toSend, toFirst, full []float64
}

func (h *hopStats) add(toSend, toFirst float64) {
	h.mu.Lock()
	h.toSend = append(h.toSend, toSend)
	h.toFirst = append(h.toFirst, toFirst)
	h.mu.Unlock()
}

func p90(v []float64) float64 {
	if len(v) == 0 {
		return -1
	}
	sort.Float64s(v)
	return v[len(v)*9/10]
}

func (h *hopStats) loop() {
	for range time.Tick(2 * time.Second) {
		h.mu.Lock()
		if len(h.toSend) > 0 {
			log.Printf("[gw.hops p90 ms] http_to_reqsent=%.1f http_to_first_audio=%.1f n=%d", p90(h.toSend), p90(h.toFirst), len(h.toSend))
		}
		h.toSend, h.toFirst = nil, nil
		h.mu.Unlock()
	}
}

var hops hopStats

type gateway struct {
	conn    net.Conn
	wmu     sync.Mutex
	streams sync.Map // uint32 -> chan []byte
	nextID  uint32
	ready   atomic.Bool
}

func (g *gateway) send(t byte, id uint32, payload []byte) error {
	buf := make([]byte, 9+len(payload))
	buf[0] = t
	binary.LittleEndian.PutUint32(buf[1:5], id)
	binary.LittleEndian.PutUint32(buf[5:9], uint32(len(payload)))
	copy(buf[9:], payload)
	g.wmu.Lock()
	defer g.wmu.Unlock()
	_, err := g.conn.Write(buf)
	return err
}

func (g *gateway) readLoop() {
	r := bufio.NewReaderSize(g.conn, 4<<20)
	hdr := make([]byte, 9)
	for {
		if _, err := io.ReadFull(r, hdr); err != nil {
			log.Fatalf("engine connection closed: %v", err)
		}
		t := hdr[0]
		id := binary.LittleEndian.Uint32(hdr[1:5])
		n := binary.LittleEndian.Uint32(hdr[5:9])
		payload := make([]byte, n)
		if _, err := io.ReadFull(r, payload); err != nil {
			log.Fatalf("engine connection closed: %v", err)
		}
		switch t {
		case tReady:
			g.ready.Store(true)
			log.Printf("engine ready")
		case tAudio:
			if ch, ok := g.streams.Load(id); ok {
				ch.(chan []byte) <- payload
			}
		case tEnd, tErr:
			if t == tErr {
				log.Printf("request %d failed: %s", id, payload)
			}
			if ch, ok := g.streams.LoadAndDelete(id); ok {
				close(ch.(chan []byte))
			}
		}
	}
}

type speechReq struct {
	Input    string `json:"input"`
	Voice    string `json:"voice"`
	Language string `json:"language"`
}

func (g *gateway) speech(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req speechReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Input == "" {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	if req.Voice == "" {
		req.Voice = "Ryan"
	}
	if req.Language == "" {
		req.Language = "English"
	}
	tArrive := time.Now()
	id := atomic.AddUint32(&g.nextID, 1)
	ch := make(chan []byte, 4096) // far more chunks than one utterance produces: the reader never blocks
	g.streams.Store(id, ch)
	body, _ := json.Marshal(req)
	if err := g.send(tReq, id, body); err != nil {
		g.streams.Delete(id)
		http.Error(w, "engine unavailable", http.StatusServiceUnavailable)
		return
	}
	tSent := time.Now()
	first := true
	h := w.Header()
	h.Set("Content-Type", "audio/pcm")
	h.Set("X-Sample-Rate", "24000")
	h.Set("X-Channels", "1")
	h.Set("X-Bit-Depth", "16")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)
	if fl != nil {
		fl.Flush()
	}
	ctx := r.Context()
	for {
		select {
		case b, ok := <-ch:
			if !ok {
				return
			}
			if first {
				first = false
				hops.add(float64(tSent.Sub(tArrive).Microseconds())/1000, float64(time.Since(tArrive).Microseconds())/1000)
			}
			if _, err := w.Write(b); err != nil {
				g.streams.Delete(id)
				_ = g.send(tCancel, id, nil)
				return
			}
			if fl != nil {
				fl.Flush()
			}
		case <-ctx.Done():
			g.streams.Delete(id)
			_ = g.send(tCancel, id, nil)
			return
		}
	}
}

// fakeSpeech streams synthetic PCM on the engine's chunk schedule (2, 4, 8, 16, 16, ... frames of
// 1920 samples, one frame every frameMs) without any engine: isolates HTTP/connection behaviour.
func fakeSpeech(frameMs int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req speechReq
		_ = json.NewDecoder(r.Body).Decode(&req)
		w.Header().Set("Content-Type", "audio/pcm")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		if fl != nil {
			fl.Flush()
		}
		sched := []int{2, 4, 8, 16}
		total, i := 90, 0
		time.Sleep(time.Duration(2*frameMs) * time.Millisecond)
		for sent := 0; sent < total; i++ {
			n := sched[min(i, len(sched)-1)]
			if sent+n > total {
				n = total - sent
			}
			buf := make([]byte, n*1920*2)
			for j := range buf {
				buf[j] = byte(j % 7)
			}
			if _, err := w.Write(buf); err != nil {
				return
			}
			if fl != nil {
				fl.Flush()
			}
			sent += n
			if sent < total {
				next := sched[min(i+1, len(sched)-1)]
				time.Sleep(time.Duration(next*frameMs) * time.Millisecond)
			}
		}
	}
}

func main() {
	sock := flag.String("sock", "/tmp/uqt.sock", "engine Unix socket")
	addr := flag.String("addr", "127.0.0.1:8091", "HTTP listen address")
	fake := flag.Int("fake", 0, "if > 0: no engine, stream synthetic audio with this many ms per frame")
	stats := flag.Bool("stats", false, "log per-hop latency percentiles every 2 s")
	flag.Parse()

	if *fake > 0 {
		mux := http.NewServeMux()
		mux.HandleFunc("/v1/audio/speech", fakeSpeech(*fake))
		mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })
		log.Printf("uqt-gw FAKE mode on %s, %d ms/frame", *addr, *fake)
		log.Fatal((&http.Server{Addr: *addr, Handler: mux}).ListenAndServe())
	}

	var conn net.Conn
	var err error
	for {
		conn, err = net.Dial("unix", *sock)
		if err == nil {
			break
		}
		time.Sleep(time.Second)
	}
	g := &gateway{conn: conn}
	go g.readLoop()
	if *stats {
		go hops.loop()
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/audio/speech", g.speech)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if g.ready.Load() {
			w.Write([]byte("ok"))
			return
		}
		http.Error(w, "warming up", http.StatusServiceUnavailable)
	})
	srv := &http.Server{Addr: *addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	log.Printf("uqt-gw listening on %s", *addr)
	log.Fatal(srv.ListenAndServe())
}
