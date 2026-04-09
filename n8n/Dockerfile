FROM docker:27-cli AS docker_cli

FROM n8nio/n8n:latest

USER root

COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER node
