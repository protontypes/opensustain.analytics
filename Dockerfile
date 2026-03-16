FROM python:3.13-slim
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -no-cache -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit-app-tab.py"]
