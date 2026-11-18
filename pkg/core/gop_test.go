package core

import (
	"sync"
	"testing"
	"time"

	"github.com/pion/rtp"
	"github.com/stretchr/testify/require"
)

func rtpPacket(ts uint32, marker bool, payload ...byte) *Packet {
	return &rtp.Packet{Header: rtp.Header{Timestamp: ts, Marker: marker}, Payload: payload}
}

func TestRTPH264Kind(t *testing.T) {
	require.Equal(t, auKey, rtpH264Kind([]byte{0x65, 0x88}))
	require.Equal(t, auParamSet, rtpH264Kind([]byte{0x67, 0x42}))
	require.Equal(t, auParamSet, rtpH264Kind([]byte{0x68, 0xce}))
	require.Equal(t, auSEI, rtpH264Kind([]byte{0x06, 0x05}))
	require.Equal(t, auOther, rtpH264Kind([]byte{0x41, 0x9a}))
	require.Equal(t, auKey, rtpH264Kind([]byte{0x7c, 0x85, 0x00}))   // FU-A start, IDR
	require.Equal(t, auOther, rtpH264Kind([]byte{0x7c, 0x81, 0x00})) // FU-A start, non-IDR
	// STAP-A with SPS + PPS
	require.Equal(t, auParamSet, rtpH264Kind([]byte{0x78, 0x00, 0x02, 0x67, 0x42, 0x00, 0x02, 0x68, 0xce}))
	// STAP-A with SPS + IDR
	require.Equal(t, auKey, rtpH264Kind([]byte{0x78, 0x00, 0x02, 0x67, 0x42, 0x00, 0x02, 0x65, 0x88}))
	// malformed STAP-A must not panic
	require.Equal(t, auOther, rtpH264Kind([]byte{0x78, 0x00, 0xff, 0x67}))
	require.Equal(t, auOther, rtpH264Kind(nil))
}

func TestRTPH265Kind(t *testing.T) {
	require.Equal(t, auKey, rtpH265Kind([]byte{19 << 1, 0x01}))      // IDR_W_RADL
	require.Equal(t, auParamSet, rtpH265Kind([]byte{32 << 1, 0x01})) // VPS
	require.Equal(t, auParamSet, rtpH265Kind([]byte{34 << 1, 0x01})) // PPS
	require.Equal(t, auSEI, rtpH265Kind([]byte{39 << 1, 0x01}))      // prefix SEI
	require.Equal(t, auOther, rtpH265Kind([]byte{1 << 1, 0x01}))     // TRAIL_R
	require.Equal(t, auKey, rtpH265Kind([]byte{49 << 1, 0x01, 0x80 | 19}))
	require.Equal(t, auOther, rtpH265Kind([]byte{49 << 1, 0x01, 0x80 | 1}))
	require.Equal(t, auParamSet, rtpH265Kind([]byte{48 << 1, 0x01, 0x00, 0x02, 32 << 1, 0x01}))
	require.Equal(t, auOther, rtpH265Kind([]byte{48 << 1, 0x01, 0x00, 0x09, 32 << 1}))
	require.Equal(t, auOther, rtpH265Kind([]byte{0x26}))
}

func TestAVCCKind(t *testing.T) {
	idr := []byte{0, 0, 0, 2, 0x65, 0x88}
	sei := []byte{0, 0, 0, 2, 0x06, 0x05}
	p := []byte{0, 0, 0, 2, 0x41, 0x9a}
	require.Equal(t, auKey, avccKind(append(append([]byte{}, sei...), idr...), false))
	require.Equal(t, auOther, avccKind(p, false))
	require.Equal(t, auKey, avccKind([]byte{0, 0, 0, 2, 19 << 1, 0x01}, true))
	require.Equal(t, auOther, avccKind([]byte{0, 0, 0, 9, 0x65}, false)) // truncated
}

type collector struct {
	Node
	got []*Packet
}

func newCollector() *collector {
	c := &collector{}
	c.Input = func(packet *Packet) { c.got = append(c.got, packet) }
	return c
}

func h264Receiver() *Receiver {
	r := NewReceiver(nil, &Codec{Name: CodecH264, PayloadType: 96})
	r.EnableGOPCache()
	return r
}

func TestReceiverGOPReplay(t *testing.T) {
	r := h264Receiver()

	p0 := rtpPacket(0, true, 0x41)            // P-frame before any keyframe: not cached
	sps := rtpPacket(3000, false, 0x67, 0x42) // keyframe access unit starts with SPS
	pps := rtpPacket(3000, false, 0x68, 0xce)
	idr := rtpPacket(3000, true, 0x65, 0x88)
	p1 := rtpPacket(6000, true, 0x41)
	for _, p := range []*Packet{p0, sps, pps, idr, p1} {
		r.Input(p)
	}

	c := newCollector()
	c.WithParent(&r.Node)
	require.Equal(t, []*Packet{sps, pps, idr, p1}, c.got, "new consumer gets the cached GOP")

	p2 := rtpPacket(9000, true, 0x41)
	r.Input(p2)
	require.Equal(t, []*Packet{sps, pps, idr, p1, p2}, c.got, "then live packets, no duplicates")

	// next keyframe resets the cache
	idr2 := rtpPacket(12000, true, 0x65, 0x88)
	r.Input(idr2)
	c2 := newCollector()
	c2.WithParent(&r.Node)
	require.Equal(t, []*Packet{idr2}, c2.got)
}

// Some cameras (e.g. Yoosee H265) send every parameter set as its own access unit with the
// marker bit set. They must stay in front of the keyframe.
func TestReceiverGOPParamSetsWithMarker(t *testing.T) {
	r := NewReceiver(nil, &Codec{Name: CodecH265, PayloadType: 96})
	r.EnableGOPCache()

	vps := rtpPacket(1000, true, 32<<1, 0x01)
	sps := rtpPacket(1000, true, 33<<1, 0x01)
	pps := rtpPacket(1000, true, 34<<1, 0x01)
	fu1 := rtpPacket(1000, false, 49<<1, 0x01, 0x80|19) // IDR first fragment
	fu2 := rtpPacket(1000, true, 49<<1, 0x01, 0x40|19)  // IDR last fragment
	p := rtpPacket(4000, true, 1<<1, 0x01)
	for _, pkt := range []*Packet{vps, sps, pps, fu1, fu2, p} {
		r.Input(pkt)
	}

	c := newCollector()
	c.WithParent(&r.Node)
	require.Equal(t, []*Packet{vps, sps, pps, fu1, fu2, p}, c.got)
}

func TestReceiverGOPParamSetsWithoutKeyframe(t *testing.T) {
	r := h264Receiver()
	r.Input(rtpPacket(1000, true, 0x67, 0x42)) // SPS alone
	r.Input(rtpPacket(2000, true, 0x41))       // followed by a P-frame: nothing to replay

	c := newCollector()
	c.WithParent(&r.Node)
	require.Empty(t, c.got)
}

func TestReceiverGOPDisabledByDefault(t *testing.T) {
	r := NewReceiver(nil, &Codec{Name: CodecH264, PayloadType: 96})
	r.Input(rtpPacket(0, true, 0x65, 0x88))
	c := newCollector()
	c.WithParent(&r.Node)
	require.Empty(t, c.got)
}

func TestReceiverGOPIgnoresAudio(t *testing.T) {
	r := NewReceiver(nil, &Codec{Name: CodecAAC, PayloadType: 97})
	r.EnableGOPCache()
	require.Nil(t, r.gop.Load())
}

func TestGOPCacheLimit(t *testing.T) {
	g := newGOPCache(&Codec{Name: CodecH264, PayloadType: 96})
	g.add(rtpPacket(0, true, 0x65))
	for i := 1; i <= gopMaxPacketsRTP; i++ {
		g.add(rtpPacket(uint32(i), true, 0x41))
	}
	require.False(t, g.valid, "too long GOP disables replay")

	c := newCollector()
	g.replay(&c.Node)
	require.Empty(t, c.got)

	g.add(rtpPacket(99999, true, 0x65))
	require.True(t, g.valid, "re-armed on the next keyframe")
}

// Viewers attaching while packets flow must get a gapless, duplicate-free sequence that starts
// at a GOP (run with -race).
func TestReceiverGOPConcurrent(t *testing.T) {
	r := h264Receiver()

	const total = 3000
	var wg sync.WaitGroup
	collectors := make([]*collector, 20)

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < total; i++ {
			payload := byte(0x41) // P-frame
			if i%30 == 0 {
				payload = 0x65 // IDR every 30 frames
			}
			p := rtpPacket(uint32(i*3000), true, payload, 0x00)
			p.SequenceNumber = uint16(i)
			r.Input(p)
		}
	}()

	for i := range collectors {
		c := newCollector()
		collectors[i] = c
		wg.Add(1)
		go func() {
			defer wg.Done()
			time.Sleep(time.Duration(i) * 200 * time.Microsecond)
			c.WithParent(&r.Node)
		}()
	}
	wg.Wait()

	for i, c := range collectors {
		if len(c.got) == 0 {
			continue
		}
		require.Equal(t, uint16(0), c.got[0].SequenceNumber%30, "collector %d must start at a keyframe", i)
		for j := 1; j < len(c.got); j++ {
			require.Equal(t, c.got[j-1].SequenceNumber+1, c.got[j].SequenceNumber, "collector %d gap or duplicate", i)
		}
	}
}
