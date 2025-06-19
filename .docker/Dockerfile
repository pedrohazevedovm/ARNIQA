# Imagem base com Python e Miniconda
FROM continuumio/miniconda3:latest

# Evita perguntas interativas durante instalação
ENV DEBIAN_FRONTEND=noninteractive

# Atualiza o sistema e instala dependências básicas
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && apt-get clean

# Cria um novo ambiente Conda
RUN conda create -n myenv python=3.10 -y

# Ativa o ambiente e instala os pacotes via Conda
RUN /bin/bash -c "source activate myenv && \
    conda install -y pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia && \
    pip install pandas==2.1.3 matplotlib==3.8.2 pyyaml==6.0.1 dotmap==1.3.30 tqdm==4.66.1 wandb==0.15.0 \
                einops==0.7.0 openpyxl==3.1.2 scikit-learn==1.3.2 scipy==1.11.4 \
                kornia==0.7.0 plotly==5.18.0 umap-learn==0.5.5"

# Ativa o ambiente por padrão
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/envs/myenv/bin:$PATH
ENV CONDA_DEFAULT_ENV=myenv

# Define diretório de trabalho
WORKDIR /workspace

# Comando padrão ao rodar o container
CMD [ "python" ]
