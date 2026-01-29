# Alterado de 'latest' para uma versão específica para garantir estabilidade
ARG DOCKER_TAG=release-v5.3
FROM espressif/idf:${DOCKER_TAG}

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Instala udev (obrigatório para gravar via USB) e usbutils (para rodar lsusb e debug)
RUN apt-get update -y && \
    apt-get install -y udev usbutils

# Configura o ambiente automaticamente ao abrir o terminal
RUN echo "source /opt/esp/idf/export.sh > /dev/null 2>&1" >> ~/.bashrc

ENTRYPOINT [ "/opt/esp/entrypoint.sh" ]
CMD ["/bin/bash", "-c"]