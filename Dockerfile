FROM python:3.13-slim

# install system requirements for OpenCV to work
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 && rm -rf /var/lib/apt/lists/*


# setup non-root user that HF spaces require
RUN useradd -m -u 1000 appuser

USER appuser

ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR $HOME/app

# copy requirements and assign ownership
COPY --chown=appuser requirements.txt .


# install dependency 
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# copy application code
COPY --chown=appuser . $HOME/app

# port 7860 to the outside HF Hub bridge
EXPOSE 7860

# Launch uvicorn on all internal adaptors (0.0.0.0) so HF can connect!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
