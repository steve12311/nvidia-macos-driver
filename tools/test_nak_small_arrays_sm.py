#!/usr/bin/env python3
"""Host test for D1: small-array registerization enabled on Ampere (SM86), Ada (SM89) and Blackwell (SM120)."""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "nvk/nvk-macos.patch"
LATE_SPLIT = ROOT / "nvk/nvk-nak-late-split.patch"
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

fails = 0


def arm(name, ok, detail=""):
    global fails
    suffix = f" ({detail})" if detail and not ok else ""
    print(("ok   " if ok else "FAIL ") + name + suffix)
    fails += not ok


def check_patch_hunks(path, target_file=None):
    lines = path.read_text().splitlines()
    cur_file = None
    i = 0
    checked = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("+++ "):
            cur_file = line[4:].split("\t")[0]
            i += 1
            continue
        m = HUNK_RE.match(line)
        if m and (target_file is None or (cur_file and cur_file.endswith(target_file))):
            old_n = int(m.group(2)) if m.group(2) is not None else 1
            new_n = int(m.group(4)) if m.group(4) is not None else 1
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
                return False, f"{path.name} hunk mismatch: expected -{old_n},+{new_n}, got -{seen_old},+{seen_new}"
            checked += 1
            continue
        i += 1
    return checked > 0, f"checked {checked} hunks"


ok, detail = check_patch_hunks(PATCH, "src/nouveau/compiler/nak_nir.c")
arm("nvk-macos.patch nak_nir.c hunks have valid line counts", ok, detail)

ok, detail = check_patch_hunks(LATE_SPLIT, "src/nouveau/compiler/nak_nir.c")
arm("nvk-nak-late-split.patch hunks have valid line counts", ok, detail)

patch_text = PATCH.read_text()
m_cond = re.search(
    r"^\+\s*if\s*\(([^\n]+nir->info\.stage == MESA_SHADER_COMPUTE)\)\s*\{\n"
    r"^\+\s*const unsigned limit = nvmtl_small_array_limit\(getenv\(\"NAK_B83_ARRAYS\"\)\);",
    patch_text,
    re.MULTILINE,
)
arm("small-array gate found in nak_postprocess_nir", m_cond is not None)

m_limit = re.search(
    r"^\+static unsigned nvmtl_small_array_limit\(const char \*value\)\n"
    r"(?:^\+.*\n)+?^\+\}",
    patch_text,
    re.MULTILINE,
)
arm("nvmtl_small_array_limit definition found in nvmtl_raw_policy.h", m_limit is not None)

if m_cond and m_limit:
    cond_expr = m_cond.group(1)
    limit_fn = "\n".join(l[1:] for l in m_limit.group(0).splitlines())
    cc = shutil.which("cc") or shutil.which("clang")
    if not cc:
        print("SKIP: needs a C compiler")
        sys.exit(2)
    harness = f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {{ MESA_SHADER_FRAGMENT = 4, MESA_SHADER_COMPUTE = 5 }};
struct fake_nak {{ unsigned sm; }};
struct fake_info {{ unsigned stage; }};
struct fake_nir {{ struct fake_info info; }};

{limit_fn}

static unsigned eval_limit(unsigned sm, unsigned stage) {{
    struct fake_nak nak_s = {{ sm }};
    struct fake_nir nir_s = {{ {{ stage }} }};
    struct fake_nak *nak = &nak_s;
    struct fake_nir *nir = &nir_s;
    if ({cond_expr}) {{
        return nvmtl_small_array_limit(getenv("NAK_B83_ARRAYS"));
    }}
    return 0;
}}

int main(void) {{
    unsetenv("NVMTL_NO_SMALL_ARRAYS");
    unsetenv("NAK_B83_ARRAYS");
    if (eval_limit(75, MESA_SHADER_COMPUTE) != 0) return 1;
    if (eval_limit(86, MESA_SHADER_COMPUTE) != 16) return 2;
    if (eval_limit(89, MESA_SHADER_COMPUTE) != 16) return 3;
    if (eval_limit(120, MESA_SHADER_COMPUTE) != 16) return 4;
    if (eval_limit(86, MESA_SHADER_FRAGMENT) != 0) return 5;
    if (eval_limit(89, MESA_SHADER_FRAGMENT) != 0) return 6;
    if (eval_limit(120, MESA_SHADER_FRAGMENT) != 0) return 7;

    setenv("NAK_B83_ARRAYS", "8", 1);
    if (eval_limit(86, MESA_SHADER_COMPUTE) != 8) return 8;
    setenv("NAK_B83_ARRAYS", "32", 1);
    if (eval_limit(89, MESA_SHADER_COMPUTE) != 32) return 9;
    setenv("NVMTL_NO_SMALL_ARRAYS", "1", 1);
    if (eval_limit(86, MESA_SHADER_COMPUTE) != 0) return 10;
    if (eval_limit(89, MESA_SHADER_COMPUTE) != 0) return 11;
    if (eval_limit(120, MESA_SHADER_COMPUTE) != 0) return 12;
    return 0;
}}
"""
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "t.c"
        exe = pathlib.Path(d) / "t"
        src.write_text(harness)
        subprocess.run([cc, "-Wall", "-Werror", "-o", str(exe), str(src)], check=True)
        res = subprocess.run([str(exe)])
        arm("extracted gate enables SM86/SM89/SM120 compute only and respects env switches", res.returncode == 0, f"rc={res.returncode}")

print("FAILED" if fails else "PASS")
sys.exit(1 if fails else 0)
