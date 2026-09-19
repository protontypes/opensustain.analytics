
cleanup:
	uv tool run pre-commit install
	uv tool run pre-commit run --all

build-json:
	python scripts/build_analytics_payloads.py

run:
	streamlit run streamlit-app-tab.py
