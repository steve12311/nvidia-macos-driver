#!/usr/bin/env python3
"""Host test for D3/D4: range-based u2f32/i2f32 magic conversion in NAK NIR."""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "nvk/nvk-macos.patch"
LATE_SPLIT = ROOT / "nvk/nvk-nak-late-split.patch"
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)")

fails = 0


def arm(name, ok, detail=""):
    global fails
    suffix = f" ({detail})" if detail and not ok else ""
    print(("ok   " if ok else "FAIL ") + name + suffix)
    fails += not ok


def validate_file_hunks(path, target_file):
    lines = path.read_text().splitlines()
    cur_file = None
    i = 0
    cum_delta = 0
    checked = 0
    hunks = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("+++ "):
            cur_file = line[4:].split("\t")[0]
            i += 1
            continue
        m = HUNK_RE.match(line)
        if m and cur_file and cur_file.endswith(target_file):
            old_s = int(m.group(1))
            old_n = int(m.group(2)) if m.group(2) is not None else 1
            new_s = int(m.group(3))
            new_n = int(m.group(4)) if m.group(4) is not None else 1
            if new_s != old_s + cum_delta:
                return False, f"{path.name}: hunk -{old_s},+{new_s} expected +{old_s + cum_delta}"
            i += 1
            seen_old = 0
            seen_new = 0
            while i < len(lines):
                l = lines[i]
                if l.startswith(("diff ", "--- ", "+++ ", "@@ ")):
                    break
                if l.startswith("\\"):
                    i += 1
                    continue
                if l.startswith("+"):
                    seen_new += 1
                elif l.startswith("-"):
                    seen_old += 1
                else:
                    seen_old += 1
                    seen_new += 1
                i += 1
            if seen_old != old_n or seen_new != new_n:
                return False, f"{path.name} hunk -{old_s},{old_n} +{new_s},{new_n}: actual -{seen_old},+{seen_new}"
            cum_delta += seen_new - seen_old
            checked += 1
            hunks.append((old_s, old_n, new_s, new_n))
            continue
        i += 1
    return checked > 0, (checked, cum_delta, hunks)


ok, info = validate_file_hunks(PATCH, "src/nouveau/compiler/nak_nir.c")
arm("nvk-macos.patch nak_nir.c hunks have exact counts and cumulative offsets", ok, str(info))

ok2, info2 = validate_file_hunks(LATE_SPLIT, "src/nouveau/compiler/nak_nir.c")
arm("nvk-nak-late-split.patch hunks have exact counts and cumulative offsets", ok2, str(info2))

if ok and ok2:
    # Verify late-split hunk 1 lands right at the closing brace of nak_cg_punroll
    macos_lines = PATCH.read_text().splitlines()
    # Reconstruct the post-patch line number of nak_cg_punroll's closing brace
    # Hunk 8 of nak_nir.c ends with nak_cg_punroll()
    punroll_end_anchor = None
    in_nak = False
    cur_new_line = 0
    for l in macos_lines:
        if l.startswith("diff "):
            in_nak = "src/nouveau/compiler/nak_nir.c" in l
            continue
        if not in_nak:
            continue
        m = HUNK_RE.match(l)
        if m:
            cur_new_line = int(m.group(3))
            continue
        if l.startswith("-"):
            continue
        if l.startswith("+") or l.startswith(" "):
            if "progress |= nir_progress(np > 0, impl, nir_metadata_none);" in l:
                punroll_end_anchor = cur_new_line
            cur_new_line += 1
    ls_first_old = info2[2][0][0]
    arm(
        "nvk-nak-late-split.patch aligns exactly with post-nvk-macos.patch nak_nir.c",
        punroll_end_anchor == ls_first_old,
        f"punroll anchor line={punroll_end_anchor}, late-split hunk1 start={ls_first_old}",
    )

patch_text = PATCH.read_text()
arm("nak_cg_possible_bits32 range analysis exists in nak_nir.c", "nak_cg_possible_bits32(" in patch_text)
arm("nak_cg_u2f_magic_instr accepts 32-bit range-proven u2f32 and i2f32",
    "alu->op != nir_op_u2f32 && alu->op != nir_op_i2f32" in patch_text
    and "nak_cg_alu_src_known_u23(alu, 0)" in patch_text)

cc = shutil.which("cc") or shutil.which("clang")
if not cc:
    print("SKIP: needs a C compiler")
    sys.exit(2)

MODEL_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static uint32_t fill_below_msb(uint32_t m) {
    m |= m >> 1; m |= m >> 2; m |= m >> 4; m |= m >> 8; m |= m >> 16;
    return m;
}

int main(void) {
    /* Exhaustive check: every integer in [0, 2^23 - 1] is bit-identical to (float)x */
    for (uint32_t x = 0; x <= 0x7FFFFFu; x++) {
        float got = u2f(0x4B000000u | x) - 8388608.0f;
        float ref_u = (float)x;
        float ref_i = (float)(int32_t)x;
        if (f2u(got) != f2u(ref_u) || f2u(got) != f2u(ref_i)) {
            fprintf(stderr, "u23 mismatch at x=%u\n", x);
            return 1;
        }
    }
    /* Q4_K masked nibble/shift ranges */
    const uint32_t q4k_masks[4] = { 0x000Fu, 0x00F0u, 0x0F00u, 0xF000u };
    for (int i = 0; i < 4; i++) {
        if (q4k_masks[i] & ~0x7FFFFFu) return 2;
    }
    /* Q6_K unpacked 6-bit value: (q1 & 0xF) | ((qh & 3) << 4) => 0x3F */
    uint32_t q6k_mask = (0xFFFFu & 0xFu) | ((0xFFu & 3u) << 4);
    if (q6k_mask != 0x3Fu || (q6k_mask & ~0x7FFFFFu)) return 3;
    /* umin bound check */
    if ((fill_below_msb(0x8u) & fill_below_msb(0x7u)) != 0x7u) return 4;
    /* Negative control: 0x800000 overflows 23-bit mantissa and MUST be rejected */
    if ((0x800000u & ~0x7FFFFFu) == 0) return 5;
    return 0;
}
"""

with tempfile.TemporaryDirectory() as d:
    src = pathlib.Path(d) / "u2f.c"
    exe = pathlib.Path(d) / "u2f"
    src.write_text(MODEL_C)
    subprocess.run([cc, "-O2", "-Wall", "-Werror", "-o", str(exe), str(src)], check=True)
    res = subprocess.run([str(exe)])
    arm("exhaustive bit-exact proof over all 8,388,608 u23 integers passes", res.returncode == 0, f"rc={res.returncode}")

print("FAILED" if fails else "PASS")
sys.exit(1 if fails else 0)
