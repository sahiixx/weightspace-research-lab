PYTHON ?= python3
.PHONY: install test demo inspect verify sweep negative-controls full-audit docker-build docker-run
install:
	$(PYTHON) -m pip install -e '.[test]'
test:
	$(PYTHON) -m pytest -q
demo:
	$(PYTHON) -m wslab pipeline --out runs/demo
inspect:
	$(PYTHON) -m wslab inspect --checkpoint runs/demo/patched.pt
verify:
	$(PYTHON) -m wslab verify --base runs/demo/base.pt --patched runs/demo/patched.pt --patch runs/demo/patch.json
sweep:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) -m wslab sweep --out runs/audit
negative-controls:
	$(PYTHON) -m wslab negative-controls --out runs/audit
full-audit:
	bash scripts/run_all.sh runs/full-audit
docker-build:
	docker compose build
docker-run:
	docker compose run --rm lab
