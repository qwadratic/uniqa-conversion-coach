"""
run_names.py — microgpt.py on names.txt with:
  • 5 random train names printed before training
  • progress timer
  • 5 generated names after training
Run from this directory (input.txt = names.txt must be present).
"""
import time, random, sys, os

# ── patch microgpt: reduce inference to 5 samples, add timing ────────────────
src = open(os.path.join(os.path.dirname(__file__), 'microgpt.py')).read()

# replace the 20-sample inference block with a 5-sample one
src = src.replace(
    'for sample_idx in range(20):',
    'for sample_idx in range(5):'
)

# Prepend timing + 5-train-sample logic (runs BEFORE the exec'd training loop)
PREAMBLE = '''
_t0_train = None
'''

src = src.replace(
    'for step in range(num_steps):',
    '_t0_train = __import__("time").time()\nfor step in range(num_steps):'
)

src = src.replace(
    'print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end=\'\\r\')',
    '''_elapsed = __import__("time").time() - _t0_train
_steps_done = step + 1
_steps_left = num_steps - _steps_done
_eta = _elapsed / _steps_done * _steps_left if _steps_done else 0
print(f"step {_steps_done:4d}/{num_steps} | loss {loss.data:.4f} | {_elapsed:.0f}s elapsed | ETA {_eta:.0f}s", end="\\r")'''
)

src = src.replace(
    'print("\\n--- inference (new, hallucinated names) ---")',
    '''print(f"\\nTraining done in {__import__("time").time()-_t0_train:.1f}s")
print("\\n--- inference (5 new, hallucinated names) ---")'''
)

# ── 5 random training examples ─────────────────────────────────────────────────
print("=" * 50)
print("5 RANDOM NAMES FROM TRAINING SET")
print("=" * 50)

# Quick load to get the docs list for display
_docs_preview = [line.strip() for line in open(
    os.path.join(os.path.dirname(__file__), 'input.txt')) if line.strip()]
random.seed(42)
random.shuffle(_docs_preview)
for i, name in enumerate(_docs_preview[:5], 1):
    print(f"  {i}. {name}")
print("=" * 50)
print()

# ── run microgpt ───────────────────────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))
exec(compile(src, 'microgpt.py', 'exec'))
