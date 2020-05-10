from pathlib import Path


root = Path(__file__).parent.parent.parent
data = root / 'data'
raw = data / 'raw'
processed = data / 'processed'
final = data / 'final'
docs = root / 'docs'
figs = root / 'figs'
models = root / 'models'
notebooks = root / 'notebooks'
src = root / 'src'
state_stats = root / 'state_stats'
