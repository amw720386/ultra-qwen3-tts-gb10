"""The launch film. Every measured number comes from results/gb10/release.json; the speech itself is illustrative.

    modal run <toolkit>/film/render_modal.py --film media/film.py --data results/gb10/release.json --out assets/film

 0.0  one grey figure speaks, its voice leaves as 80 ms frames
 5.2  cut into the head: 28 Talker layers, 15 predictor steps, the codec, then the mouth
12.0  pull out: 128 of them
16.4  three crowds of 128 press play together, in real time
23.0  the result
"""
import math

import kit
import numpy as np
from mathutils import Vector

DURATION = 27.0
S = {}
COLS, ROWS = 8, 16
DX, DY = 2.15, 1.25
HERO = (2, 0)                                   # the figure of the first shot, in the crowd of the third
RACE = 16.4
FRAME = 0.08                                    # seconds of audio per frame
N_VAR = 10
WALLS = 28
CYCLE, UNIT = 2.4, 3.1                         # seconds per frame made inside the head; its timeline is written over UNIT
SINK = (-0.80, -0.50)
YAW = 0.72                                      # the layer walls turn towards the camera
RING, RR = (-0.005, -0.10, 0.185), 0.062       # the predictor's 15 steps, behind the forehead


def crowd_pos(c, r):
    return np.array([(c - HERO[0]) * DX, (r - HERO[1]) * DY, 0.0])


def build(assets, data):
    import bpy
    sc = kit.stage()
    S.update(sc=sc, cam=sc.camera, mouth=kit.mouth(assets), data=data)
    S["clay"], S["hero_mat"] = kit.clay("clay", sink=SINK), kit.clay("hero", clip=True, sink=SINK)
    S["metal"] = kit.clay("metal", rough=0.42)
    S["blue"], S["white"], S["amber"] = kit.flat("blue", kit.BLUE, 1.0), kit.flat("white", kit.WHITE), kit.flat("amber", kit.ORANGE)
    S["hero"] = kit.bust(assets, "hero", S["hero_mat"], subdiv=2)
    S["hero"].color = (.21, .215, .225, 1)
    # crowd meshes: a few speaking variants, instanced
    S["var"] = [kit.bust(assets, f"var{i}", S["clay"], subdiv=1) for i in range(N_VAR)]
    S["mute"] = kit.bust(assets, "mute", S["clay"], subdiv=1)
    for o in S["var"] + [S["mute"]]:
        o.location = (0, 0, -300)
    rng = np.random.default_rng(4)

    def inst(src, name):
        o = src.copy()
        sc.collection.objects.link(o)
        o.name = name
        return o
    S["crowd"] = []
    for r in range(ROWS):
        for c in range(COLS):
            if (c, r) == HERO:
                continue
            v = int(rng.integers(N_VAR))
            S["crowd"].append((inst(S["var"][v], f"c{c}_{r}"), crowd_pos(c, r), v))
    # the race: three blocks, each figure in two copies (speaking and silent), one hidden at a time
    S["blocks"] = []
    ang = 0.0
    for b in range(3):
        rows = []
        for r in range(32):
            for c in range(4):
                p = np.array([(c - 1.5) * 1.62 + (b - 1) * 8.5, r * 1.2, 0.0])
                v = int(rng.integers(N_VAR))
                rows.append((inst(S["var"][v], f"r{b}_{c}_{r}_on"), inst(S["mute"], f"r{b}_{c}_{r}_off"), p, v))
        S["blocks"].append(rows)
    S["ang"] = ang
    # voices
    S["plates"] = kit.Boxes("plates", 6000, S["blue"])
    # inside the head
    S["walls"], S["hot"] = [], []
    T = 9
    for i in range(WALLS):
        x = -0.595 + i * 0.0160
        r = math.sqrt(max(0.15, 1 - ((x + 0.34) / 0.36) ** 2))
        h, w = 0.215 * r, 0.15 * r
        cs, ss, hc, hs = [], [], [], []
        for a in range(T):
            for b in range(T):
                q, z = (a - (T - 1) / 2) / T * 2 * w, 0.19 + (b - (T - 1) / 2) / T * 2 * h
                sz = (0.0035, 2 * w / T * 0.80, 2 * h / T * 0.80)
                (hc if rng.random() < 0.42 else cs).append((x + q * math.sin(-YAW), 0.03 + q * math.cos(YAW), z))
                (hs if len(hc) > len(hs) else ss).append(sz)
        v, f = kit.boxes_mesh(cs, ss, YAW)
        S["walls"].append(kit.mesh_object(f"w{i}", v, f.tolist(), S["metal"]))
        v, f = kit.boxes_mesh(hc, hs, YAW)
        S["hot"].append((kit.mesh_object(f"h{i}", v, f.tolist(), S["blue"]), kit.mesh_object(f"g{i}", v, f.tolist(), S["metal"])))
    S["ring"] = []
    for k in range(15):
        a = math.pi / 2 - 2 * math.pi * k / 15
        c = (RING[0] + RR * math.cos(a), RING[1], RING[2] + RR * math.sin(a))
        v, f = kit.boxes_mesh([c], [(0.019, 0.04, 0.019)], -0.0)
        S["ring"].append((kit.mesh_object(f"rg{k}", v, f.tolist(), S["metal"]), kit.mesh_object(f"rb{k}", v, f.tolist(), S["amber"])))
    S["codec"] = []
    for k in range(3):
        c = (-0.01 + 0.010 * k, -0.01, 0.055 - 0.062 * k)
        v, f = kit.boxes_mesh([c], [(0.075 + 0.02 * k, 0.06, 0.036)])
        S["codec"].append((kit.mesh_object(f"k{k}", v, f.tolist(), S["metal"]), kit.mesh_object(f"kb{k}", v, f.tolist(), S["blue"])))
    v, f = kit.boxes_mesh([(0, 0, 0)], [(0.016, 0.016, 0.016)])
    S["token"] = kit.mesh_object("token", v, f.tolist(), S["white"])
    S["inside"] = S["walls"] + [o for p in S["hot"] + S["ring"] + S["codec"] for o in p] + [S["token"]]
    # light: rims from behind, almost nothing in front
    S["L"] = {
        "face": kit.area("face", (1.7, 2.0, 0.5), (0.0, 0, -0.1), 1.0, 60, 2.4),
        "back": kit.area("back", (-2.2, 2.4, 1.3), (0, 0, 0), 1.6, 70, 2.4),
        "top": kit.area("top", (0.2, 1.2, 3.2), (0, 0, 0), 1.4, 34),
        "fill": kit.area("fill", (0.4, -4.5, 1.2), (0, 0, 0), 4, 7),
        "in": kit.area("in", (-0.5, -1.5, 1.2), (-0.25, 0, 0.15), 0.5, 0),
        "crowd": kit.area("crowd", (8, 34, 17), (6, 8, 0), 34, 0, 12),
        "crowdf": kit.area("crowdf", (4, -22, 24), (6, 4, 0), 20, 0),
        "race": kit.area("race", (0, 52, 24), (0, 8, 0), 60, 0, 14),
        "racef": kit.area("racef", (0, -24, 26), (0, 6, 0), 40, 0),
    }
    return sc


def ribbon(out, pos, ang, t, seed, t_start, length, n_max, scale=1.0, fade=1.0):
    """Append one figure's voice: an 80 ms plate every 80 ms, leaving the mouth along its gaze."""
    m = S["mouth"]
    ca, sa = math.cos(ang), math.sin(ang)
    now = t - t_start
    k1 = int(now / FRAME)
    for k in range(max(0, k1 - n_max + 1), k1 + 1):
        age = now - k * FRAME
        x = m[0] + 0.05 + 0.62 * age * scale
        if x - m[0] > length:
            continue
        e = kit.voice(k * FRAME, seed)
        grow = kit.ease(age / 0.14) * (1 - kit.ease((x - m[0] - length + 0.22) / 0.22)) * fade
        if grow < 0.02:
            continue
        h = (0.018 + 0.36 * e) * grow
        out[0].append((pos[0] + ca * x - sa * m[1], pos[1] + sa * x + ca * m[1], pos[2] + m[2] + 0.02))
        out[1].append((0.024, 0.024, h))
        out[2].append(ang)


def reveal(t, p):
    """The crowd comes out of the dark from the hero outwards."""
    return kit.ramp(t, 12.15 + 0.055 * math.hypot(p[0], p[1]), 0.7)


def show(o, on):
    o.hide_render = not on


def frame(t):
    cam, L = S["cam"], S["L"]
    out = ([], [], [])
    anchors = {}
    race = t >= RACE
    inside = 5.1 < t < 12.1
    crowd_on = 12.0 < t < RACE
    for o in S["inside"]:
        show(o, inside)
    show(S["hero"], not race)
    for o, _, _ in S["crowd"]:
        show(o, crowd_on)
    for b, rows in enumerate(S["blocks"]):
        for on, off, _, _ in rows:
            show(on, False)
            show(off, False)
    for i, o in enumerate(S["var"]):
        kit.speak(o.data.shape_keys.key_blocks, t, 0.7 + i * 0.83)
    for k in L:
        L[k].data.energy = 0

    if not race:
        # the cut: a plane sweeps away the near half of the head, and later brings it back
        cut = kit.ramp(t, 5.2, 1.4) * (1 - kit.ramp(t, 11.2, 1.0))
        S["hero_mat"].node_tree.nodes["CLIP"].outputs[0].default_value = kit.lerp(-0.62, -0.005, cut) if cut > 0 else -99
        keys = S["hero"].data.shape_keys.key_blocks
        slow = kit.ramp(t, 5.0, 0.6) * (1 - kit.ramp(t, 11.9, 0.6))
        kit.speak(keys, t, 0.0, 1 - 0.85 * slow)
        pull = kit.ramp(t, 12.0, 3.4)
        ribbon(out, (0, 0, 0), 0.0, t, 0.0, -0.9, kit.lerp(2.9, 1.15, kit.ramp(t, 11.8, 0.8)), 60, fade=1 - slow)
        # inside the head: one frame of speech is made, slowly
        if inside:
            u = (t - 6.5) / CYCLE
            k, u = math.floor(u), (u % 1.0) * UNIT
            live = 6.5 <= t < 6.5 + 2 * CYCLE
            in_walls, in_ring, in_codec = u / 1.35, (u - 1.45) / 0.8, (u - 2.35) / 0.45
            for i in range(WALLS):
                d = in_walls * WALLS - i
                lit = live and 0 <= d
                g = max(0.0, 1 - max(0.0, d - 1.5) / 9.0) if lit else 0.0
                on, off = S["hot"][i]
                show(on, lit and g > 0.02)
                show(off, not (lit and g > 0.02))
                on.color = (max(g, .12),) * 3 + (1,)
                off.color = S["walls"][i].color = (.30, .31, .325, 1)
            for j, (off, on) in enumerate(S["ring"]):
                lit = live and 0 <= in_ring and in_ring * 15 >= j and u < 2.9
                show(on, lit)
                show(off, not lit)
                off.color = (.2, .205, .215, 1)
                on.color = (1, 1, 1, 1)
            for j, (off, on) in enumerate(S["codec"]):
                lit = live and in_codec * 3 >= j and u < 3.05
                show(on, lit)
                show(off, not lit)
                off.color = (.2, .205, .215, 1)
                on.color = (1, 1, 1, 1)
            # the token's path
            pts = [(0.0, (-0.62, -0.12, 0.19)), (1.35, (-0.12, -0.12, 0.19)), (1.47, (RING[0], RING[1] - 0.04, RING[2] + RR))]
            if u < 1.47:
                p = kit.vlerp(pts[0][1], pts[1][1], u / 1.35) if u < 1.35 else kit.vlerp(pts[1][1], pts[2][1], (u - 1.35) / 0.12)
            elif u < 2.25:
                a = math.pi / 2 - 2 * math.pi * (u - 1.45) / 0.8
                p = Vector((RING[0] + RR * math.cos(a), RING[1] - 0.04, RING[2] + RR * math.sin(a)))
            else:
                p = kit.vlerp((RING[0], RING[1] - 0.06, RING[2] + RR), (S["mouth"][0], -0.06, S["mouth"][2]), kit.clamp((u - 2.25) / 0.55))
            S["token"].location = p
            show(S["token"], live and u < 2.8)
            # and the frame it became leaves the mouth
            for q in range(0, k + 1):
                age = t - (6.5 + (q + 2.78 / UNIT) * CYCLE)
                if 0 <= age < 3.0 and t < 12.2:
                    x = S["mouth"][0] + 0.05 + 0.34 * age
                    out[0].append((x, 0, S["mouth"][2] + 0.02))
                    out[1].append((0.03, 0.03, 0.30 * kit.ease(age / 0.25) * (1 - kit.ramp(t, 11.6, 0.5))))
                    out[2].append(0.0)
        for o, p, v in S["crowd"]:
            o.location = p
            o.color = (.19, .195, .205, reveal(t, p))
        if crowd_on:
            for i, (o, p, v) in enumerate(S["crowd"]):
                ribbon(out, p, 0.0, t, 0.7 + v * 0.83, -i * 0.013, 1.15, 24, fade=reveal(t, p))
        # light
        near = 1 - kit.ramp(t, 11.7, 1.0)
        L["face"].data.energy, L["back"].data.energy, L["top"].data.energy, L["fill"].data.energy = 62 * near, 74 * near, 34 * near, 6 * near
        L["in"].data.energy = 20 * cut
        L["crowd"].data.energy, L["crowdf"].data.energy = 10500 * kit.ramp(t, 11.6, 1.6), 900 * kit.ramp(t, 11.6, 1.6)
        # camera
        loc = kit.path(t, [(0, (0.74, -5.5, 0.06)), (5.0, (0.56, -4.7, 0.02)), (7.4, (0.02, -2.95, 0.20)), (11.3, (0.50, -2.75, 0.28)), (13.0, (1.2, -5.2, 1.3)), (16.4, (10.2, -13.5, 7.6))])
        at = kit.path(t, [(0, (0.66, 0, -0.14)), (5.0, (0.50, 0, -0.14)), (7.4, (-0.10, 0, -0.02)), (11.3, (-0.04, 0, -0.03)), (13.0, (0.8, 0.4, -0.05)), (16.4, (5.2, 7.2, -0.9))])
        kit.look(cam, loc, at)
        cam.data.lens = kit.lerp(62, 42, pull)
    else:
        S["hero_mat"].node_tree.nodes["CLIP"].outputs[0].default_value = -99
        u = t - RACE
        starts = starts_of(S["data"])
        end = kit.ramp(t, 22.4, 1.0)
        for b, rows in enumerate(S["blocks"]):
            talking = u >= starts[b]
            since = u - starts[b]
            g = kit.ease(since / 0.18) if talking else 0.0
            for i, (on, off, p, v) in enumerate(rows):
                o = on if talking else off
                show(o, True)
                o.location = p
                o.rotation_euler = (0, 0, S["ang"])
                s = kit.lerp(0.011, 0.26, g)
                o.color = (s, s * 1.02, s * 1.06, 1)
                if talking:
                    ribbon(out, p, S["ang"], t, 0.7 + v * 0.83, RACE + starts[b], 0.80, 18)
        dim = 1 - 0.975 * end
        L["race"].data.energy, L["racef"].data.energy = 30000 * dim, 6500 * dim
        S["blue"].node_tree.nodes["Emission"].inputs[1].default_value = dim
        loc = kit.path(t, [(RACE, (-0.5, -13.4, 5.9)), (27.0, (0.5, -12.4, 5.3))])
        kit.look(cam, loc, (0, 6.0, -1.35))
        cam.data.lens = 22.5
        for b in range(3):
            anchors[f"b{b}"] = ((b - 1) * 8.5 + 0.2, -1.0, -1.15)
    if not race:
        S["blue"].node_tree.nodes["Emission"].inputs[1].default_value = 1
    S["plates"].set(*out)
    return anchors


def starts_of(data):
    g = {r["id"]: r for r in data["head_to_head"]}
    return [g[f"{k}-128"]["heard_p50_ms"] / 1000 for k in ("uqt", "sglang", "vllm")]


# --- overlay (runs after rendering, without Blender) ---
def _e(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _fade(t, a, b, d=0.4):
    return _e((t - a) / d) * (1 - _e((t - b + d) / d))


POSTER = 3.0
CLIPS = {"voice": (0.0, 5.0, 720), "inside": (6.5, 11.3, 720), "crowd": (12.0, 16.35, 720), "race": (16.4, 22.2, 720)}


def overlay(t, pts, data):
    g = {r["id"]: r for r in data["head_to_head"]}
    ms = {r["weights"]: r["ms_per_frame"] for r in data["single_stream"]}["FP8 + FP4"]
    out = []

    def line(a, b, text, sub=None, d=0.4):
        k = _fade(t, a, b, d) if a > 0 else 1 - _e((t - b + d) / d)
        out.append({"text": text, "x": 0.047, "y": 0.875, "size": 62, "a": k})
        if sub:
            out.append({"text": sub, "x": 0.048, "y": 0.928, "size": 27, "weight": "Regular", "color": "#8b97a8", "a": k})
    line(0.0, 2.3, "One voice.")
    line(2.3, 5.2, f"{ms:g} ms of work for every 80 ms of speech.")
    line(6.3, 7.7, "28 layers.", d=0.3)
    line(7.7, 8.6, "15 steps.", d=0.25)
    line(8.6, 11.3, "One frame of speech.")
    line(13.0, 14.9, "128 voices.")
    line(14.9, 16.4, "One GB10.", d=0.3)
    R = 16.4
    if t >= R:
        u = t - R
        k = 1 - _e((t - 22.2) / 0.5)
        for b, (key, name) in enumerate((("uqt", "Ours"), ("sglang", "sglang-omni"), ("vllm", "vLLM-Omni"))):
            s = g[f"{key}-128"]["heard_p50_ms"] / 1000
            x, y = pts.get(f"b{b}", (0.2 + 0.3 * b, 0.8))
            done = u >= s
            out.append({"text": name, "x": x, "y": y, "size": 34, "weight": "Regular", "anchor": "c", "a": k, "color": "#e8edf4" if done else "#8b97a8"})
            out.append({"text": f"{min(u, s):.2f} s", "x": x, "y": y + 0.056, "size": 58, "anchor": "c", "a": k, "tnum": True, "color": "#3b82f6" if done and b == 0 else "#e8edf4" if done else "#8b97a8"})
        out.append({"text": "128 people press play. When does the voice start?", "x": 0.5, "y": 0.085, "size": 46, "anchor": "c", "a": k * (1.0 if u > 0.0 else 0)})
        out.append({"text": "Median time until the voice is heard, 128 clients on one GB10, shown in real time", "x": 0.5, "y": 0.962, "size": 24, "weight": "Regular", "anchor": "c", "a": k, "color": "#8b97a8"})
        e = _e((t - 22.9) / 0.7)
        v = data["throughput_vs"]["sglang"]["geomean"]
        out.append({"text": f"{v:.1f}× sglang-omni's throughput.", "x": 0.5, "y": 0.40, "size": 96, "anchor": "c", "a": e})
        out.append({"text": "Qwen3-TTS 1.7B on one GB10, geometric mean from 1 to 256 clients", "x": 0.5, "y": 0.49, "size": 29, "weight": "Regular", "anchor": "c", "a": _e((t - 23.4) / 0.6), "color": "#a3afc0"})
        out.append({"text": "github.com/amw720386/ultra-qwen3-tts-gb10", "x": 0.5, "y": 0.575, "size": 31, "weight": "Regular", "anchor": "c", "a": _e((t - 23.9) / 0.6)})
        out.append({"svg": "magni-lockup-on-dark.svg", "x": 0.5, "y": 0.80, "w": 250, "a": _e((t - 24.6) / 0.7)})
    return out
