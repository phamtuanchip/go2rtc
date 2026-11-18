package core

import "encoding/binary"

// GOP cache limits. They keep a replay below the Sender queue sizes (4096 for RTP video,
// 64 for non-RTP video), so a replayed GOP is never dropped by a new consumer.
const (
	gopMaxBytes      = 8 << 20
	gopMaxPacketsRTP = 2048
	gopMaxPacketsRaw = 48
)

// kinds of access unit starts, as seen by the GOP cache
const (
	auOther    = iota // regular (non key) picture
	auParamSet        // VPS / SPS / PPS
	auSEI             // supplemental enhancement information
	auKey             // IDR / IRAP picture
)

// gopCache keeps the video packets of the latest group of pictures (GOP), starting from
// the parameter sets and keyframe. A consumer that joins a running stream gets this GOP first
// and can start decoding at once instead of waiting up to a full keyframe interval.
type gopCache struct {
	h265 bool
	rtp  bool

	packets []*Packet
	bytes   int
	valid   bool // packets start with a keyframe (or its parameter sets) and fit the limits
	prefix  bool // packets are only parameter sets / SEI waiting for the keyframe

	prevTS     uint32
	prevMarker bool
	started    bool
}

func newGOPCache(codec *Codec) *gopCache {
	return &gopCache{h265: codec.Name == CodecH265, rtp: codec.IsRTP()}
}

func (g *gopCache) reset(valid bool) {
	g.packets = g.packets[:0]
	g.bytes = 0
	g.valid = valid
	g.prefix = false
}

func (g *gopCache) add(packet *Packet) {
	if g.isAUStart(packet) {
		switch g.kind(packet) {
		case auParamSet:
			// Some cameras send VPS/SPS/PPS as separate access units (with the marker bit)
			// right before the keyframe: keep them as the start of the next GOP.
			if !g.prefix {
				g.reset(true)
				g.prefix = true
			}
		case auSEI:
			if !g.prefix {
				g.endPrefix()
			}
		case auKey:
			if !g.prefix {
				g.reset(true)
			}
			g.prefix = false
		default:
			g.endPrefix()
		}
	} else if g.prefix && g.kind(packet) == auKey {
		// parameter sets and the keyframe in one access unit (same timestamp, no marker)
		g.prefix = false
	}

	if g.valid {
		maxPackets := gopMaxPacketsRaw
		if g.rtp {
			maxPackets = gopMaxPacketsRTP
		}
		if len(g.packets) >= maxPackets || g.bytes+len(packet.Payload) > gopMaxBytes {
			// GOP too long to replay safely: behave as if there was no cache until the next keyframe
			g.packets = nil
			g.reset(false)
		} else {
			g.packets = append(g.packets, packet)
			g.bytes += len(packet.Payload)
		}
	}

	g.prevTS = packet.Timestamp
	g.prevMarker = packet.Marker
	g.started = true
}

// endPrefix handles a regular picture: parameter sets not followed by a keyframe are useless.
func (g *gopCache) endPrefix() {
	if g.prefix {
		g.reset(false)
	}
}

func (g *gopCache) replay(child *Node) {
	if !g.valid || g.prefix || child.NoGOPReplay {
		return
	}
	for _, packet := range g.packets {
		child.Input(packet)
	}
}

// isAUStart reports whether the packet begins a new access unit.
func (g *gopCache) isAUStart(packet *Packet) bool {
	if !g.rtp {
		return true // non-RTP codecs carry one whole access unit (AVCC) per packet
	}
	// RTP: a new access unit starts after the marker bit, or when the timestamp changes
	return !g.started || g.prevMarker || packet.Timestamp != g.prevTS
}

func (g *gopCache) kind(packet *Packet) int {
	switch {
	case !g.rtp:
		return avccKind(packet.Payload, g.h265)
	case g.h265:
		return rtpH265Kind(packet.Payload)
	default:
		return rtpH264Kind(packet.Payload)
	}
}

func kindH264(nalType byte) int {
	switch nalType {
	case 5:
		return auKey
	case 7, 8:
		return auParamSet
	case 6:
		return auSEI
	}
	return auOther
}

func kindH265(nalType byte) int {
	switch {
	case nalType >= 16 && nalType <= 21:
		return auKey
	case nalType >= 32 && nalType <= 34:
		return auParamSet
	case nalType == 39 || nalType == 40:
		return auSEI
	}
	return auOther
}

// maxKind returns the most important kind (key > param set > SEI > other) of aggregated units.
func maxKind(a, b int) int {
	if b > a {
		return b
	}
	return a
}

// rtpH264Kind classifies an RTP H264 payload (RFC 6184): single NAL unit, STAP-A, FU-A.
func rtpH264Kind(b []byte) int {
	if len(b) < 1 {
		return auOther
	}
	switch nalType := b[0] & 0x1F; nalType {
	case 24: // STAP-A
		kind := auOther
		for i := 1; i+2 < len(b); {
			size := int(binary.BigEndian.Uint16(b[i:]))
			if size == 0 || i+2+size > len(b) {
				break
			}
			kind = maxKind(kind, kindH264(b[i+2]&0x1F))
			i += 2 + size
		}
		return kind
	case 28: // FU-A: the type is in the FU header
		if len(b) < 2 {
			return auOther
		}
		return kindH264(b[1] & 0x1F)
	default:
		return kindH264(nalType)
	}
}

// rtpH265Kind classifies an RTP H265 payload (RFC 7798): single NAL unit, AP, FU.
func rtpH265Kind(b []byte) int {
	if len(b) < 2 {
		return auOther
	}
	switch nalType := (b[0] >> 1) & 0x3F; nalType {
	case 48: // aggregation packet
		kind := auOther
		for i := 2; i+2 < len(b); {
			size := int(binary.BigEndian.Uint16(b[i:]))
			if size == 0 || i+2+size > len(b) {
				break
			}
			kind = maxKind(kind, kindH265((b[i+2]>>1)&0x3F))
			i += 2 + size
		}
		return kind
	case 49: // fragmentation unit: the type is in the FU header
		if len(b) < 3 {
			return auOther
		}
		return kindH265(b[2] & 0x3F)
	default:
		return kindH265(nalType)
	}
}

// avccKind classifies a length-prefixed (AVCC) access unit.
func avccKind(b []byte, h265 bool) int {
	kind := auOther
	for i := 0; i+4 < len(b); {
		size := int(binary.BigEndian.Uint32(b[i:]))
		if size == 0 || i+4+size > len(b) {
			break
		}
		if h265 {
			kind = maxKind(kind, kindH265((b[i+4]>>1)&0x3F))
		} else {
			kind = maxKind(kind, kindH264(b[i+4]&0x1F))
		}
		i += 4 + size
	}
	return kind
}
