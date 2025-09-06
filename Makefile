.PHONY: run update clean

# Regenerate figures
run:
	python3 "Yield Curve PCA.py"

# Regenerate, commit, and push latest figures
update: run
	git add reports/figures/curves_latest.png reports/figures/curves_*.png README.md || true
	git commit -m "Update figures: $$(date -u +%Y-%m-%dT%H:%M:%SZ)" || echo "No changes to commit."
	git push

clean:
	rm -f reports/figures/curves_latest.png
