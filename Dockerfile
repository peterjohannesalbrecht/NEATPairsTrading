# Specify base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy current direc into the container
COPY . .

# Install packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]