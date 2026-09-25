"""Build the Japanese demo video (VOICEVOX narration + Pillow frames + ffmpeg).

Reproducible re-run recipe for docs/demo_video_ja.mp4. Deterministic given
the VOICEVOX server: narration text drives frame timing (audio-first, the
#1 A/V sync rule); frame reveal is a clean typing effect; the final mux
never calls a model/server.

Setup:
    pip install -e ".[demo]"        # Pillow (VOICEVOX server + ffmpeg are system tools)
    VOICEVOX: ~/.voicevox/squashfs-root/vv-engine/run --host 127.0.0.1 --port 50021

Usage:
    .venv/bin/python scripts/build_demo.py            # build docs/demo_video_ja.mp4
    .venv/bin/python scripts/build_demo.py --check    # verify an existing build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT_MP4 = REPO / "docs" / "demo_video_ja.mp4"
VOICEVOX = "http://127.0.0.1:50021"
SPEAKER = 2  # 四国めたん
FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"

W, H = 1280, 720
FPS = 30

# (scene_id, narration, title, [(role, text), ...]) — role colors the line.
# Narration is spoken; title is the frame heading (short — the title area
# does not wrap); text is what the frame shows (the demo's real stdout).
SCENES: list[tuple[str, str, str, list[tuple[str, str]]]] = [
    ("hook", "これは架空の日本の銀行の質問応答アシスタントです。検索、計算ツール、ガードレール、そして動作を証明する評価セットを備えた、誠実なLLMエンジニアリングの実演です。", "finance-ai-assistant — 架空銀行FAQアシスタント", [
        ("cmd", "$ python main.py \"質問\""),
        ("answer", "  -> 出典付きの回答 [FAQ-013]"),
        ("note", "RAG + tool + guardrails + eval suite"),
    ]),
    ("rag", "まず通常の質問です。回答は資料に基づき、出典が付きます。検索結果も見てみましょう。", "Demo 1 — RAG回答と検索結果", [
        ("cmd", "$ python main.py \"住宅ローンの繰上返済の手数料を教えてください。店頭で申し込む場合も。\" --show-context"),
        ("context", "検索: [FAQ-013] ほっと住宅ローン : 83.235"),
        ("context", "      [FAQ-014] ほっと住宅ローン : 23.184"),
        ("answer", "UNTRUSTED_CONTENT_BEGIN"),
        ("answer", "繰上返済の手数料は以下の通りです。"),
        ("answer", "- インターネットバンキング: 無料"),
        ("answer", "- 店頭窓口: 8,800円（税込）"),
        ("cite", "[FAQ-013]"),
        ("answer", "UNTRUSTED_CONTENT_END"),
        ("note", "検索文書は untrusted マーカーで分離（OWASP LLM01対策）"),
    ]),
    ("tool", "計算が必要な質問では、ツールを使って正確に計算します。", "Demo 2 — ツール実行", [
        ("cmd", "$ python main.py \"3000万円を年利2.1%で35年間借りた場合、月々の返済額はいくらですか？\""),
        ("tool_use", "ツール実行: calculate_monthly_payment(principal=30000000, rate=0.021, years=35)"),
        ("answer", "月々の返済額は 100,925円 です。"),
        ("cite", "[FAQ-011] [FAQ-012]"),
        ("note", "ツール入力は検証済み（非有限数・bool・小数年を拒否）"),
    ]),
    ("guardrail", "投資助言にあたる質問は、LLMを呼ばずに決定的に拒否します。", "Demo 3 — ガードレール", [
        ("cmd", "$ python main.py \"おすすめの投資信託を教えてください\""),
        ("warning", "申し訳ありませんが、この質問にはお答えできません。"),
        ("warning", "「個別の商品選択・売買など投資助言にあたる質問」にあたるためです。"),
        ("note", "ガードレール作動 (investment_advice): LLM呼び出しなし"),
    ]),
    ("eval", "最後に評価です。コーパスと質問を固定し、変更の劣化を検出する回帰評価です。", "Eval — 回帰評価", [
        ("cmd", "$ python -m evals.run_eval"),
        ("answer", "retrieval MRR 0.903 / recall@5 100% / precision@1 84%"),
        ("answer", "answer 5.0 / refuse 100% / cite 100%"),
        ("note", "45 tests pass / gitleaks clean / CI green (3.11-3.13)"),
    ]),
]

ROLE_COLORS = {
    "title": (120, 220, 255),   # cyan
    "cmd": (120, 255, 120),     # green
    "answer": (240, 240, 240),  # white
    "cite": (120, 220, 255),    # cyan
    "warning": (255, 190, 90),  # amber
    "context": (170, 170, 170), # gray
    "tool_use": (255, 220, 120),# amber
    "note": (150, 150, 150),    # gray
}


def synth_clip(text: str, cache_dir: Path) -> tuple[Path, float]:
    """Synthesize one narration clip; returns (wav_path, duration_sec)."""
    key = hashlib.sha256(f"{SPEAKER}:{text}".encode()).hexdigest()[:16]
    wav = cache_dir / f"{key}.wav"
    if wav.exists():
        dur = float(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(wav)],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return wav, dur
    q_url = f"{VOICEVOX}/audio_query?text={urllib.parse.quote(text)}&speaker={SPEAKER}"
    with urllib.request.urlopen(urllib.request.Request(q_url, method="POST"), timeout=60) as r:
        query = json.loads(r.read())
    with urllib.request.urlopen(
        urllib.request.Request(
            f"{VOICEVOX}/synthesis?speaker={SPEAKER}",
            data=json.dumps(query).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        ),
        timeout=120,
    ) as r:
        wav.write_bytes(r.read())
    dur = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(wav)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )
    return wav, dur


def wrap_line(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """Soft-wrap Japanese text at the content width (hard clipping is the failure mode)."""
    lines, cur = [], ""
    for ch in text:
        if font.getlength(cur + ch) > max_w and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def render_frame(visible_lines: list[tuple[str, str]], title: str) -> Image.Image:
    img = Image.new("RGB", (W, H), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.truetype(FONT_PATH, 26)
    font_body = ImageFont.truetype(FONT_PATH, 22)
    margin = 60
    y = 50
    draw.text((margin, y), title, font=font_title, fill=(220, 220, 220))
    draw.line([(margin, y + 44), (W - margin, y + 44)], fill=(80, 80, 90), width=2)
    y += 70
    max_w = W - 2 * margin
    for role, text in visible_lines:
        color = ROLE_COLORS.get(role, (240, 240, 240))
        for ln in wrap_line(text, font_body, max_w):
            draw.text((margin, y), ln, font=font_body, fill=color)
            y += 32
        if role in ("cmd", "tool_use"):
            y += 10
    return img


def build() -> Path:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cache_dir = REPO / "docs" / "demo_captures" / "voice_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Synthesize narration; audio-first timing.
        clips = []
        for sid, narration, _title, lines in SCENES:
            wav, dur = synth_clip(narration, cache_dir)
            clips.append((sid, narration, lines, wav, dur + 0.6))  # post_pad 0.6s
            print(f"scene {sid}: {dur:.1f}s narration")

        # 2. Render frames: per scene, a typing effect over the clip duration.
        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        n_total = 0
        timeline = []  # (start_sec, dur, wav)
        t = 0.0
        for sid, narration, lines, wav, dur in clips:
            n_frames = max(1, int(round(dur * FPS)))
            title = next(s[2] for s in SCENES if s[0] == sid)
            total_chars = sum(len(t2) for _, t2 in lines)
            for i in range(n_frames):
                progress = (i + 1) / n_frames
                target = int(progress * total_chars)
                visible, consumed = [], 0
                for role, text in lines:
                    if consumed >= target:
                        break
                    take = min(len(text), target - consumed)
                    visible.append((role, text[:take]))
                    consumed += take
                # Unrevealed lines stay hidden (no fill loop): revealing
                # later lines before the typing reaches them breaks the
                # effect's ordering promise.
                frame = render_frame(visible, title)
                frame.save(frames_dir / f"f{n_total:05d}.png")
                n_total += 1
            timeline.append((t, dur, wav))
            t += dur
        print(f"rendered {n_total} frames")

        # 3. Concatenate audio with the same scene durations the frames use:
        # pad each clip with 0.6s of silence (post_pad) before concat —
        # ffmpeg concatenates without gaps, so an unpadded concat makes
        # audio drift 0.6s earlier per scene and -shortest then truncates
        # the final scene's tail.
        concat_list = tmp / "wav_list.txt"
        padded_dir = tmp / "padded"
        padded_dir.mkdir()
        with concat_list.open("w") as f:
            for _, dur, wav in timeline:
                padded = padded_dir / wav.name
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                     "-af", "apad=pad_dur=0.6", "-ar", "44100",
                     str(padded)],
                    capture_output=True, check=True,
                )
                f.write(f"file '{padded}'\n")
        wav_out = tmp / "narration.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c", "copy", str(wav_out)],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(frames_dir / "f%05d.png"),
             "-i", str(wav_out),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-c:a", "aac", "-shortest",
             str(OUT_MP4)],
            capture_output=True, check=True,
        )
    print(f"built {OUT_MP4} ({OUT_MP4.stat().st_size / 1e6:.1f} MB)")
    return OUT_MP4


def check() -> bool:
    """Programmatic verification (ffprobe), per the skill's machine-led rule."""
    ok = True
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_name,width,height,duration", "-of", "json", str(OUT_MP4)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(p.stdout)
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" or s.get("codec_name") in ("h264", "hevc")), streams[0] if streams else None)
    a = next((s for s in streams if s.get("codec_name") == "aac"), None)
    print(f"video: {v.get('codec_name')} {v.get('width')}x{v.get('height')}" if v else "video: MISSING")
    print(f"audio: {a.get('codec_name')}" if a else "audio: MISSING")
    if not a:
        ok = False
    if v and (v.get("width"), v.get("height")) != (1280, 720):
        ok = False
    # Duration gate: a stale 3m12s file with the right codecs must not pass.
    # The README documents ~38s; accept 30–60s as the target band.
    dur_s = None
    try:
        dur_s = float(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(OUT_MP4)],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, ValueError):
        pass
    if dur_s is None or not (30.0 <= dur_s <= 60.0):
        print(f"duration: {dur_s}s OUTSIDE 30-60s target band")
        ok = False
    else:
        print(f"duration: {dur_s:.1f}s (in band)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify existing build only")
    args = parser.parse_args()
    if args.check:
        return 0 if check() else 1
    build()
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
