
cleanup:
	uv tool run pre-commit install
	uv tool run pre-commit run --all

run:
	streamlit run streamlit-app-tab.py
