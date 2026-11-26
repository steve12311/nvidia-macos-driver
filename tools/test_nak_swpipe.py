#!/usr/bin/env python3
"""Host test for D6: 1-stage load software pipelining (nak_cg_swpipe) for large counted compute loops."""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "nvk/nvk-macos.patch"
LATE_SPLIT = ROOT / "nvk/nvk-nak-late-split.patch"
POST_TEST = ROOT / "nvk/nvk-nak-post-test-swpipe.patch"
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

ok3, info3 = validate_file_hunks(POST_TEST, "src/nouveau/compiler/nak_nir.c")
arm("post-test pipeline patch has exact hunk counts and cumulative offsets", ok3, str(info3))

patch_text = PATCH.read_text()
arm("nak_cg_swpipe exists with NVMTL_NAK_NO_SWPIPE kill switch and scoreboard load cap",
    "nak_cg_swpipe(nir_shader *nir)" in patch_text
    and 'getenv("NVMTL_NAK_NO_SWPIPE")' in patch_text
    and 'getenv("NVMTL_NAK_SWPIPE_LOADS")' in patch_text
    and "OPT(nir, nak_cg_swpipe)" in patch_text)
arm("nak_cg_swpipe runs after vectorizer/predload and respects raw_pressure",
    "if (!raw_pressure && OPT(nir, nak_cg_swpipe))" in patch_text
    and patch_text.index("nir_opt_load_store_vectorize") < patch_text.index("OPT(nir, nak_cg_swpipe)"))

cc = shutil.which("cc") or shutil.which("clang")
if not cc:
    print("SKIP: needs a C compiler")
    sys.exit(2)

MODEL_C = r"""
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static int oob_accesses = 0;

static float safe_pred_load(const float *buf, int len, int idx, bool pred) {
    if (!pred)
        return 0.0f;
    if (idx < 0 || idx >= len) {
        oob_accesses++;
        return -999999.0f;
    }
    return buf[idx];
}

/* Reference loop: for (int iv = init_iv; iv < n; iv += step) acc += a[iv] * b[iv]; */
static float run_ref_signed(const float *a, const float *b, int init_iv, int n, int step, int *final_iv) {
    float acc = 1.25f;
    int iv = init_iv;
    while (iv < n) {
        acc += a[iv] * b[iv];
        iv += step;
    }
    *final_iv = iv;
    return acc;
}

/* 1-stage predicated software-pipelined loop mirroring nak_cg_swpipe_apply */
static float run_swpipe_signed(const float *a, const float *b, int len, int init_iv, int n, int step, int *final_iv) {
    float acc = 1.25f;
    int iv = init_iv;
    if (iv < n) {
        float va = safe_pred_load(a, len, iv, true);
        float vb = safe_pred_load(b, len, iv, true);
        for (;;) {
            int iv_next = (int)((uint32_t)iv + (uint32_t)step);
            bool has_next = ((uint32_t)n - (uint32_t)iv) > (uint32_t)step;
            float next_a = safe_pred_load(a, len, iv_next, has_next);
            float next_b = safe_pred_load(b, len, iv_next, has_next);
            acc += va * vb;
            iv = iv_next;
            if (!has_next)
                break;
            va = next_a;
            vb = next_b;
        }
    }
    *final_iv = iv;
    return acc;
}

/* Overflow edge case check for (uint32_t)(n - iv) > step on signed/unsigned boundaries */
static int check_overflow_bounds(void) {
    /* Signed near INT_MAX: iv = INT_MAX - 2, n = INT_MAX, step = 4 => exactly 1 trip, has_next MUST be false */
    int32_t n = INT32_MAX;
    int32_t iv = INT32_MAX - 2;
    uint32_t step = 4u;
    if (((uint32_t)n - (uint32_t)iv) > step)
        return 1;
    /* Signed negative to positive: iv = -3, n = 5, step = 4 => next iv = 1 < 5, has_next MUST be true */
    if (!(((uint32_t)5 - (uint32_t)(-3)) > 4u))
        return 2;
    /* Unsigned near UINT_MAX: iv = UINT_MAX - 2, n = UINT_MAX, step = 4 => has_next MUST be false */
    if ((UINT32_MAX - (UINT32_MAX - 2u)) > 4u)
        return 3;
    return 0;
}

static bool post_test_continue(uint32_t next, uint32_t end, bool signed_iv) {
    uint32_t sign_bit = signed_iv ? UINT32_C(1) << 31 : 0;
    return (next ^ sign_bit) < (end ^ sign_bit);
}

static float run_post_test(const float *buf, int len, uint32_t end) {
    uint32_t iv = 0, ptr = 1;
    float value = safe_pred_load(buf, len, ptr, true), sum = 0;
    for (;;) {
        iv += 4;
        ptr += 3;
        bool has_next = post_test_continue(iv, end, false);
        float prefetched = safe_pred_load(buf, len, ptr, has_next);
        sum += value;
        if (!has_next)
            return sum;
        value = prefetched;
    }
}

static int check_post_test(const float *buf) {
    float expected = 0;
    for (uint32_t trips = 1; trips <= 20; trips++) {
        int len = (int)(trips * 3 - 1);
        expected += buf[len - 1];
        if (run_post_test(buf, len, trips * 4) != expected || oob_accesses)
            return 4;
    }
    if (!post_test_continue((uint32_t)INT32_MAX + 2u, INT32_MAX, true) ||
        !post_test_continue((UINT32_MAX - 1u) + 4u, UINT32_MAX, false))
        return 5;
    return 0;
}

int main(void) {
    float a[64], b[64];
    for (int i = 0; i < 64; i++) {
        a[i] = (float)(i + 1) * 0.5f;
        b[i] = (float)(64 - i) * 0.25f;
    }
    const int steps[] = { 1, 2, 4, 7 };
    for (size_t si = 0; si < sizeof(steps) / sizeof(steps[0]); si++) {
        int step = steps[si];
        for (int init_iv = 0; init_iv <= 8; init_iv++) {
            for (int n = 0; n <= 32; n++) {
                int ref_iv = 0, got_iv = 0;
                float ref = run_ref_signed(a, b, init_iv, n, step, &ref_iv);
                float got = run_swpipe_signed(a, b, n, init_iv, n, step, &got_iv);
                if (ref != got || ref_iv != got_iv || oob_accesses != 0) {
                    fprintf(stderr, "mismatch step=%d init=%d n=%d ref=%f(%d) got=%f(%d) oob=%d\n",
                            step, init_iv, n, ref, ref_iv, got, got_iv, oob_accesses);
                    return 1;
                }
            }
        }
    }
    int rc = check_overflow_bounds();
    return rc ? rc : check_post_test(a);
}
"""

with tempfile.TemporaryDirectory() as d:
    src = pathlib.Path(d) / "swpipe.c"
    exe = pathlib.Path(d) / "swpipe"
    src.write_text(MODEL_C)
    subprocess.run([cc, "-O2", "-Wall", "-Werror", "-o", str(exe), str(src)], check=True)
    res = subprocess.run([str(exe)])
    arm("C models cover pre/post-tested loops, independent strides, suppressed final loads and wraparound",
        res.returncode == 0, f"rc={res.returncode}")

print("FAILED" if fails else "PASS")
sys.exit(1 if fails else 0)
