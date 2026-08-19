.PHONY: smoke test

# Offline smoke: runs two representative experiments end to end on CPU,
# sklearn built-in data, no downloads. Prints metrics and asserts a floor.
smoke:
	python scripts/smoke.py

# Same experiments driven through pytest.
test:
	python -m pytest tests/ -v
