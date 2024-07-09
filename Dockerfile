FROM apache/airflow:2.9.2

# Copiar archivos y crear directorio como root
USER root
COPY requirements.txt /requirements.txt
RUN mkdir -p /home/airflow/.kaggle
COPY kaggle.json /home/airflow/.kaggle/kaggle.json

# Establecer permisos correctos para el archivo kaggle.json
RUN chmod 600 /home/airflow/.kaggle/kaggle.json

# Volver a usuario airflow para el resto de las operaciones
USER airflow

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt